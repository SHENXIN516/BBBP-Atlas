from yacs.config import CfgNode as CN


def set_cfg(cfg):

    cfg.dataset = 'ZINC'
    cfg.num_workers = 8
    cfg.device = 0
    cfg.handtune = ''
    cfg.seed = None
    cfg.downsample = False
    cfg.version = 'final'
    cfg.task = -1

    cfg.train = CN()
    cfg.train.batch_size = 128
    cfg.train.epochs = 100
    cfg.train.runs = 3
    cfg.train.lr = 0.001
    cfg.train.lr_patience = 50
    cfg.train.lr_decay = 0.5
    cfg.train.wd = 0.
    cfg.train.dropout = 0.

    cfg.model = CN()
    cfg.model.gnn_type = 'GINEConv' 
    cfg.model.hidden_size = 128
    cfg.model.num_layers = 4
    cfg.model.mini_layers = 0
    cfg.model.pool = 'add'
    cfg.model.residual = True
    cfg.model.virtual_node = False
    cfg.model.hops_dim = 16
    cfg.model.embs = (0, 1, 2)  
    cfg.model.embs_combine_mode = 'concat'
    cfg.model.mlp_layers = 1
    cfg.model.use_normal_gnn = False

    cfg.subgraph = CN()
    cfg.subgraph.hops = 3
    cfg.subgraph.walk_length = 0

    cfg.subgraph.walk_p = 1.0
    cfg.subgraph.walk_q = 1.0
    cfg.subgraph.walk_repeat = 5

    cfg.subgraph.online = True 

    cfg.sampling = CN()
    cfg.sampling.mode = None 
    cfg.sampling.redundancy = 0
    cfg.sampling.stride = 2
    cfg.sampling.random_rate = 0.5
    cfg.sampling.batch_factor = 1

    return cfg


import os
import argparse

def update_cfg(cfg, args_str=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="", metavar="FILE", help="Path to config file")
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")

    if isinstance(args_str, str):
        args = parser.parse_args(args_str.split())
    else:
        args = parser.parse_args()
    cfg = cfg.clone()

    if os.path.isfile(args.config):
        cfg.merge_from_file(args.config)

    cfg.merge_from_list(args.opts)

    return cfg

cfg = set_cfg(CN())