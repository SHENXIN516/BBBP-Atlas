import torch as th
import torch.nn.functional as F
import numpy as np
from torch_scatter import scatter_add
from torch_geometric.utils import to_dense_batch
from torch import nn
from torch_geometric.nn import global_mean_pool
from .layer.subgraph import subgraph
from .transform_AK.transform import SubgraphsTransform
from .transform_AK.config import cfg, update_cfg
import math
import logging
import math
from torch.distributions import Normal

from .transform_AK.element import MLP, DiscreteEncoder, Identity, VNUpdate


def glorot_orthogonal(tensor, scale):
    if tensor is not None:
        th.nn.init.orthogonal_(tensor.data)
        scale /= ((tensor.size(-2) + tensor.size(-1)) * tensor.var())
        tensor.data *= scale.sqrt()

class MultiHeadAttentionLayer(nn.Module):

    def __init__(self, num_input_feats, num_output_feats,
                 num_heads, using_bias=False, update_edge_feats=True, update_coords=False):
        super(MultiHeadAttentionLayer, self).__init__()

        self.num_output_feats = num_output_feats
        self.num_heads = num_heads
        self.using_bias = using_bias
        self.update_edge_feats = update_edge_feats
        self.update_coords = update_coords

        self.Q = nn.Linear(num_input_feats, self.num_output_feats * self.num_heads, bias=using_bias)
        self.K = nn.Linear(num_input_feats, self.num_output_feats * self.num_heads, bias=using_bias)
        self.V = nn.Linear(num_input_feats, self.num_output_feats * self.num_heads, bias=using_bias)
        self.edge_feats_projection = nn.Linear(num_input_feats, self.num_output_feats * self.num_heads,
                                               bias=using_bias)

        self.reset_parameters()

    def reset_parameters(self):
        scale = 2.0
        if self.using_bias:
            glorot_orthogonal(self.Q.weight, scale=scale)
            self.Q.bias.data.fill_(0)

            glorot_orthogonal(self.K.weight, scale=scale)
            self.K.bias.data.fill_(0)

            glorot_orthogonal(self.V.weight, scale=scale)
            self.V.bias.data.fill_(0)

            glorot_orthogonal(self.edge_feats_projection.weight, scale=scale)
            self.edge_feats_projection.bias.data.fill_(0)
        else:
            glorot_orthogonal(self.Q.weight, scale=scale)
            glorot_orthogonal(self.K.weight, scale=scale)
            glorot_orthogonal(self.V.weight, scale=scale)
            glorot_orthogonal(self.edge_feats_projection.weight, scale=scale)

    def propagate_attention(self, edge_index, node_feats_q, node_feats_k, node_feats_v, edge_feats_projection, coords):
        row, col = edge_index
        e_out = None
        alpha = node_feats_k[row] * node_feats_q[col]
        alpha = (alpha / np.sqrt(self.num_output_feats)).clamp(-5.0, 5.0)
        alpha = alpha * edge_feats_projection
        if self.update_edge_feats:
            e_out = alpha

        alphax = th.exp((alpha.sum(-1, keepdim=True)).clamp(-5.0, 5.0))
        wV = scatter_add(node_feats_v[row] * alphax, col, dim=0, dim_size=node_feats_q.size(0))
        z = scatter_add(alphax, col, dim=0, dim_size=node_feats_q.size(0))
        return wV, z, e_out, coords

    def forward(self, x, edge_attr, edge_index, coords):
        row, col = edge_index
        node_feats_q = self.Q(x).view(-1, self.num_heads, self.num_output_feats)
        node_feats_k = self.K(x).view(-1, self.num_heads, self.num_output_feats)
        node_feats_v = self.V(x).view(-1, self.num_heads, self.num_output_feats)
        edge_feats_projection = self.edge_feats_projection(edge_attr).view(-1, self.num_heads, self.num_output_feats)
        wV, z, e_out, coords = self.propagate_attention(edge_index, node_feats_q, node_feats_k, node_feats_v,
                                                        edge_feats_projection, coords)

        h_out = wV / (z + th.full_like(z, 1e-6))

        return h_out, e_out, coords


class GraphTransformerModule(nn.Module):

    def __init__(
            self,
            num_hidden_channels,
            activ_fn=nn.SiLU(),
            residual=True,
            num_attention_heads=4,
            norm_to_apply='batch',
            dropout_rate=0.1,
            num_layers=4,
    ):
        super(GraphTransformerModule, self).__init__()

        self.activ_fn = activ_fn
        self.residual = residual
        self.num_attention_heads = num_attention_heads
        self.norm_to_apply = norm_to_apply
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers

        self.apply_layer_norm = 'layer' in self.norm_to_apply.lower()

        self.num_hidden_channels, self.num_output_feats = num_hidden_channels, num_hidden_channels
        if self.apply_layer_norm:
            self.layer_norm1_node_feats = nn.LayerNorm(self.num_output_feats)
            self.layer_norm1_edge_feats = nn.LayerNorm(self.num_output_feats)
        else:  
            self.batch_norm1_node_feats = nn.BatchNorm1d(self.num_output_feats)
            self.batch_norm1_edge_feats = nn.BatchNorm1d(self.num_output_feats)

        self.mha_module = MultiHeadAttentionLayer(
            self.num_hidden_channels,
            self.num_output_feats // self.num_attention_heads,
            self.num_attention_heads,
            self.num_hidden_channels != self.num_output_feats, 
            update_edge_feats=True
        )

        self.O_node_feats = nn.Linear(self.num_output_feats, self.num_output_feats)
        self.O_edge_feats = nn.Linear(self.num_output_feats, self.num_output_feats)

        dropout = nn.Dropout(p=self.dropout_rate) if self.dropout_rate > 0.0 else nn.Identity()
        self.node_feats_MLP = nn.ModuleList([
            nn.Linear(self.num_output_feats, self.num_output_feats * 2, bias=False),
            self.activ_fn,
            dropout,
            nn.Linear(self.num_output_feats * 2, self.num_output_feats, bias=False)
        ])

        if self.apply_layer_norm:
            self.layer_norm2_node_feats = nn.LayerNorm(self.num_output_feats)
            self.layer_norm2_edge_feats = nn.LayerNorm(self.num_output_feats)
        else:  
            self.batch_norm2_node_feats = nn.BatchNorm1d(self.num_output_feats)
            self.batch_norm2_edge_feats = nn.BatchNorm1d(self.num_output_feats)

        self.edge_feats_MLP = nn.ModuleList([
            nn.Linear(self.num_output_feats, self.num_output_feats * 2, bias=False),
            self.activ_fn,
            dropout,
            nn.Linear(self.num_output_feats * 2, self.num_output_feats, bias=False)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        scale = 2.0
        glorot_orthogonal(self.O_node_feats.weight, scale=scale)
        self.O_node_feats.bias.data.fill_(0)
        glorot_orthogonal(self.O_edge_feats.weight, scale=scale)
        self.O_edge_feats.bias.data.fill_(0)

        for layer in self.node_feats_MLP:
            if hasattr(layer, 'weight'):  
                glorot_orthogonal(layer.weight, scale=scale)

        for layer in self.edge_feats_MLP:
            if hasattr(layer, 'weight'):
                glorot_orthogonal(layer.weight, scale=scale)

    def run_gt_layer(self, data, node_feats, edge_feats):
        """Perform a forward pass of geometric attention using a multi-head attention (MHA) module."""
        node_feats_in1 = node_feats  
        edge_feats_in1 = edge_feats  

        if self.apply_layer_norm:
            node_feats = self.layer_norm1_node_feats(node_feats)
            edge_feats = self.layer_norm1_edge_feats(edge_feats)
        else:  
            node_feats = self.batch_norm1_node_feats(node_feats)
            edge_feats = self.batch_norm1_edge_feats(edge_feats)

        node_attn_out, edge_attn_out, data.pos = self.mha_module(node_feats, edge_feats, data.edge_index, data.pos)

        node_feats = node_attn_out.view(-1, self.num_output_feats)
        edge_feats = edge_attn_out.view(-1, self.num_output_feats)

        node_feats = F.dropout(node_feats, self.dropout_rate, training=self.training)
        edge_feats = F.dropout(edge_feats, self.dropout_rate, training=self.training)

        node_feats = self.O_node_feats(node_feats)
        edge_feats = self.O_edge_feats(edge_feats)

        if self.residual:
            node_feats = node_feats_in1 + node_feats  
            edge_feats = edge_feats_in1 + edge_feats  

        node_feats_in2 = node_feats  
        edge_feats_in2 = edge_feats  

        if self.apply_layer_norm:
            node_feats = self.layer_norm2_node_feats(node_feats)
            edge_feats = self.layer_norm2_edge_feats(edge_feats)
        else:  
            node_feats = self.batch_norm2_node_feats(node_feats)
            edge_feats = self.batch_norm2_edge_feats(edge_feats)

        for layer in self.node_feats_MLP:
            node_feats = layer(node_feats)
        for layer in self.edge_feats_MLP:
            edge_feats = layer(edge_feats)

        if self.residual:
            node_feats = node_feats_in2 + node_feats  
            edge_feats = edge_feats_in2 + edge_feats  

        return node_feats, edge_feats

    def forward(self, data, node_feats, edge_feats):
        node_feats, edge_feats = self.run_gt_layer(data, node_feats, edge_feats)
        return node_feats, edge_feats


class FinalGraphTransformerModule(nn.Module):

    def __init__(self,
                 num_hidden_channels,
                 activ_fn=nn.SiLU(),
                 residual=True,
                 num_attention_heads=4,
                 norm_to_apply='batch',
                 dropout_rate=0.1,
                 num_layers=4):
        super(FinalGraphTransformerModule, self).__init__()

        self.activ_fn = activ_fn
        self.residual = residual
        self.num_attention_heads = num_attention_heads
        self.norm_to_apply = norm_to_apply
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers

        self.apply_layer_norm = 'layer' in self.norm_to_apply.lower()

        self.num_hidden_channels, self.num_output_feats = num_hidden_channels, num_hidden_channels
        if self.apply_layer_norm:
            self.layer_norm1_node_feats = nn.LayerNorm(self.num_output_feats)
            self.layer_norm1_edge_feats = nn.LayerNorm(self.num_output_feats)
        else:  
            self.batch_norm1_node_feats = nn.BatchNorm1d(self.num_output_feats)
            self.batch_norm1_edge_feats = nn.BatchNorm1d(self.num_output_feats)

        self.mha_module = MultiHeadAttentionLayer(
            self.num_hidden_channels,
            self.num_output_feats // self.num_attention_heads,
            self.num_attention_heads,
            self.num_hidden_channels != self.num_output_feats,  
            update_edge_feats=False)

        self.O_node_feats = nn.Linear(self.num_output_feats, self.num_output_feats)

        dropout = nn.Dropout(p=self.dropout_rate) if self.dropout_rate > 0.0 else nn.Identity()
        self.node_feats_MLP = nn.ModuleList([
            nn.Linear(self.num_output_feats, self.num_output_feats * 2, bias=False),
            self.activ_fn,
            dropout,
            nn.Linear(self.num_output_feats * 2, self.num_output_feats, bias=False)
        ])

        if self.apply_layer_norm:
            self.layer_norm2_node_feats = nn.LayerNorm(self.num_output_feats)
        else:  
            self.batch_norm2_node_feats = nn.BatchNorm1d(self.num_output_feats)

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        scale = 2.0
        glorot_orthogonal(self.O_node_feats.weight, scale=scale)
        self.O_node_feats.bias.data.fill_(0)

        for layer in self.node_feats_MLP:
            if hasattr(layer, 'weight'):  
                glorot_orthogonal(layer.weight, scale=scale)

    def run_gt_layer(self, data, node_feats, edge_feats):
        node_feats_in1 = node_feats 

        if self.apply_layer_norm:
            node_feats = self.layer_norm1_node_feats(node_feats)
            edge_feats = self.layer_norm1_edge_feats(edge_feats)
        else:  
            node_feats = self.batch_norm1_node_feats(node_feats)
            edge_feats = self.batch_norm1_edge_feats(edge_feats)

        node_attn_out, _, data.pos = self.mha_module(node_feats, edge_feats, data.edge_index, data.pos)
        node_feats = node_attn_out.view(-1, self.num_output_feats)
        node_feats = F.dropout(node_feats, self.dropout_rate, training=self.training)
        node_feats = self.O_node_feats(node_feats)

        if self.residual:
            node_feats = node_feats_in1 + node_feats  

        node_feats_in2 = node_feats  

        if self.apply_layer_norm:
            node_feats = self.layer_norm2_node_feats(node_feats)
        else:  
            node_feats = self.batch_norm2_node_feats(node_feats)

        for layer in self.node_feats_MLP:
            node_feats = layer(node_feats)

        if self.residual:
            node_feats = node_feats_in2 + node_feats  

        return node_feats

    def forward(self, data, node_feats, edge_feats):
        node_feats = self.run_gt_layer(data, node_feats, edge_feats)
        return node_feats

class GraphTransformer(nn.Module):

    def __init__(
            self,
            in_channels,
            edge_features=10,
            num_hidden_channels=128,
            activ_fn=nn.SiLU(),
            transformer_residual=True,
            num_attention_heads=4,
            norm_to_apply='batch',
            dropout_rate=0.1,
            num_layers=4,
            **kwargs
    ):
        super(GraphTransformer, self).__init__()

        self.activ_fn = activ_fn
        self.transformer_residual = transformer_residual
        self.num_attention_heads = num_attention_heads
        self.norm_to_apply = norm_to_apply
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers

        self.node_encoder = nn.Linear(in_channels, num_hidden_channels)
        self.edge_encoder = nn.Linear(edge_features, num_hidden_channels)

        num_intermediate_layers = max(0, num_layers - 1)
        gt_block_modules = [GraphTransformerModule(
            num_hidden_channels=num_hidden_channels,
            activ_fn=activ_fn,
            residual=transformer_residual,
            num_attention_heads=num_attention_heads,
            norm_to_apply=norm_to_apply,
            dropout_rate=dropout_rate,
            num_layers=num_layers) for _ in range(num_intermediate_layers)]
        if num_layers > 0:
            gt_block_modules.extend([FinalGraphTransformerModule(
                num_hidden_channels=num_hidden_channels,
                activ_fn=activ_fn,
                residual=transformer_residual,
                num_attention_heads=num_attention_heads,
                norm_to_apply=norm_to_apply,
                dropout_rate=dropout_rate,
                num_layers=num_layers)])
        self.gt_block = nn.ModuleList(gt_block_modules)

        self.readout_layer = nn.Sequential(
            nn.Linear(num_hidden_channels, num_hidden_channels // 2),
            nn.SiLU(),
            nn.Linear(num_hidden_channels // 2, 2),
        )

    def forward(self, data):
        node_feats = self.node_encoder(data.x)
        edge_feats = self.edge_encoder(data.edge_attr)

        for gt_layer in self.gt_block[:-1]:
            node_feats, edge_feats = gt_layer(data, node_feats, edge_feats)

        node_feats = self.gt_block[-1](data, node_feats, edge_feats)

        prop = global_mean_pool(node_feats, data.batch)

        cls = self.readout_layer(prop)

        return cls


class SubGT(nn.Module):
    def __init__(self,
                 in_channels,
                 edge_features=10,
                 num_hidden_channels=128,
                 activ_fn=nn.SiLU(),
                 transformer_residual=True,
                 num_attention_heads=4,
                 norm_to_apply='batch',
                 dropout_rate=0.1,
                 num_layers=4,
                 **kwargs
                 ):
        super(SubGT, self).__init__()

        self.activ_fn = activ_fn
        self.transformer_residual = transformer_residual
        self.num_attention_heads = num_attention_heads
        self.norm_to_apply = norm_to_apply
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers

        self.node_encoder = nn.Linear(in_channels, num_hidden_channels)
        self.edge_encoder = nn.Linear(edge_features, num_hidden_channels)

        num_intermediate_layers = max(0, num_layers - 1)
        gt_block_modules = [GraphTransformerModule(
            num_hidden_channels=num_hidden_channels,
            activ_fn=activ_fn,
            residual=transformer_residual,
            num_attention_heads=num_attention_heads,
            norm_to_apply=norm_to_apply,
            dropout_rate=dropout_rate,
            num_layers=num_layers) for _ in range(num_intermediate_layers)]
        if num_layers > 0:
            gt_block_modules.extend([FinalGraphTransformerModule(
                num_hidden_channels=num_hidden_channels,
                activ_fn=activ_fn,
                residual=transformer_residual,
                num_attention_heads=num_attention_heads,
                norm_to_apply=norm_to_apply,
                dropout_rate=dropout_rate,
                num_layers=num_layers)])
        self.gt_block = nn.ModuleList(gt_block_modules)

        self.transform = SubgraphsTransform(cfg.subgraph.hops,
                                            walk_length=cfg.subgraph.walk_length,
                                            p=cfg.subgraph.walk_p,
                                            q=cfg.subgraph.walk_q,
                                            repeat=cfg.subgraph.walk_repeat,
                                            sampling_mode=cfg.sampling.mode,
                                            minimum_redundancy=cfg.sampling.redundancy,
                                            shortest_path_mode_stride=cfg.sampling.stride,
                                            random_mode_sampling_rate=cfg.sampling.random_rate,
                                            random_init=True)

        self.transform_eval = SubgraphsTransform(cfg.subgraph.hops,
                                                 walk_length=cfg.subgraph.walk_length,
                                                 p=cfg.subgraph.walk_p,
                                                 q=cfg.subgraph.walk_q,
                                                 repeat=cfg.subgraph.walk_repeat,
                                                 sampling_mode=None,
                                                 random_init=False)

        self.sub_GT = subgraph(self.gt_block, num_hidden_channels)

        self.readout_layer = nn.Sequential(
            nn.Linear(num_hidden_channels, num_hidden_channels // 2),
            nn.SiLU(),
            nn.Linear(num_hidden_channels // 2, 2),
        )

    def forward(self, data, model_type=None):
        edge_index = data.edge_index
        data.x = self.node_encoder(data.x)
        data.edge_attr = self.edge_encoder(data.edge_attr)

        sample = dict()

        sample['subgraphs_batch'], sample['subgraphs_nodes_mapper'], sample['subgraphs_edges_mapper'], sample[
            'combined_subgraphs'], sample['hop_indicator'], sample['num_nodes'] = self.transform_eval(data.edge_index,
                                                                                                      data.x.size()[0])

        node_feats = self.sub_GT(data, sample, model_type)

        prop = global_mean_pool(node_feats, data.batch)

        cls = self.readout_layer(prop)

        return cls

