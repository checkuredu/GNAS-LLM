from nas_bench_graph import light_read as lightread
from nas_bench_graph import Arch
import numpy as np
link_list = [
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 1],
        [0, 0, 1, 2],
        [0, 0, 1, 3],
        [0, 1, 1, 1],
        [0, 1, 1, 2],
        [0, 1, 2, 2],
        [0, 1, 2, 3]
    ]
gnn_list = [
    "gat",  # GAT with 2 heads 0
    "gcn",  # GCN 1
    "gin",  # GIN 2
    "cheb",  # chebnet 3
    "sage",  # sage 4
    "arma",     #   5
    "graph",  # k-GNN 6
    "fc",  # fully-connected 7
    "skip"  # skip connection 8
]
def best_link(dataname):
    if(dataname == 'pubmed'):
        linknum = 2
    elif(dataname == 'arxiv'):
        linknum = 8
    elif(dataname == 'cora'):
        linknum = 7
    elif (dataname == 'citeseer'):
        linknum= 6
    return link_list[linknum]

def give_bench(dataname,link):
    bench = lightread(dataname)
    keys = list(bench.keys())
    rank = np.array([bench[k]['perf'] for k in keys]).argsort()[::-1].argsort()
    for k, r in zip(keys, rank):
        bench[k]['rank'] = int(r + 1)
    valid_rank = np.array([bench[k]['valid_perf'] for k in keys]).argsort()[::-1].argsort()
    for k, r in zip(keys, valid_rank):
        bench[k]['rank_test'] = int(r + 1)
    rank_list = []
    for i in gnn_list:
        for j in gnn_list:
            for k in gnn_list:
                for l in gnn_list:
                    if i == 'skip' and j == 'skip' and k == 'skip' and l == 'skip':
                        continue
                    arch = Arch(link, [i, j, k, l])
                    if arch.check_isomorph():
                        rank_list.append(arch.valid_hash())
    rank_only = np.array([bench[k]['perf'] for k in rank_list]).argsort()[::-1].argsort()
    for k, r in zip(rank_list, rank_only):
        bench[k]['rank_oper'] = int(r + 1)
    rank_valid = np.array([bench[k]['valid_perf'] for k in rank_list]).argsort()[::-1].argsort()
    for k, r in zip(keys, rank_valid):
        bench[k]['rank_oper_test'] = int(r + 1)
    return bench

