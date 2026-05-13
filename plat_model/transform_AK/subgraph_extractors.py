import torch
from torch_sparse import SparseTensor 

def k_hop_subgraph(edge_index, num_nodes, num_hops):
    row, col = edge_index.to(torch.long)

    sparse_adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
    hop_masks = [torch.eye(num_nodes, dtype=torch.bool, device=edge_index.device)] 
    hop_indicator = row.new_full((num_nodes, num_nodes), -1)
    hop_indicator[hop_masks[0]] = 0
    for i in range(num_hops):
        next_mask = sparse_adj.matmul(hop_masks[i].float()) > 0
        hop_masks.append(next_mask)
        hop_indicator[(hop_indicator==-1) & next_mask] = i+1
    hop_indicator = hop_indicator.T  
    node_mask = (hop_indicator >= 0) 
    return node_mask, hop_indicator


from torch_cluster import random_walk
def random_walk_subgraph(edge_index, num_nodes, walk_length, p=1, q=1, repeat=1, cal_hops=True, max_hops=10):
    row, col = edge_index
    start = torch.arange(num_nodes, device=edge_index.device)
    walks = [random_walk(row, col,
                         start=start,
                         walk_length=walk_length,
                         p=p, q=q,
                         num_nodes=num_nodes) for _ in range(repeat)]
    walk = torch.cat(walks, dim=-1)
    node_mask = row.new_empty((num_nodes, num_nodes), dtype=torch.bool)
    node_mask.fill_(False)
    node_mask[start.repeat_interleave((walk_length+1)*repeat), walk.reshape(-1)] = True
    if cal_hops: 
        sparse_adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
        hop_masks = [torch.eye(num_nodes, dtype=torch.bool, device=edge_index.device)]
        hop_indicator = row.new_full((num_nodes, num_nodes), -1)
        hop_indicator[hop_masks[0]] = 0
        for i in range(max_hops):
            next_mask = sparse_adj.matmul(hop_masks[i].float())>0
            hop_masks.append(next_mask)
            hop_indicator[(hop_indicator==-1) & next_mask] = i+1
            if hop_indicator[node_mask].min() != -1:
                break
        return node_mask, hop_indicator
    return node_mask, None

from torch_sparse import mul
def ppr_topk(edge_index, num_nodes, k=10, alpha=0.1, t=5):
    start = torch.eye(num_nodes, dtype=torch.float)
    sparse_adj = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(num_nodes, num_nodes))

    deg_inv = sparse_adj.sum(-1).pow(-1)
    deg_inv[torch.isinf(deg_inv)] = 0
    sparse_adj = mul(sparse_adj, deg_inv.view(-1, 1))
    ppr = start 
    for _ in range(t):
        ppr = (1-alpha)*sparse_adj.matmul(ppr) + alpha*start
    _, node_idx = torch.topk(ppr, k, dim=-1)

    node_mask = node_idx.new_empty((num_nodes, num_nodes), dtype=torch.bool)
    node_mask.fill_(False)
    node_mask[torch.arange(num_nodes).repeat_interleave(k), node_idx.reshape(-1)] = True
    return node_mask, None
