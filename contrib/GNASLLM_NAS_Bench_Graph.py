import torch
from nas_bench_graph import light_read as lightread
from nas_bench_graph import Arch
from base.base import BaseTrainer,BaseSeachSpace
import json
import requests
import time
import numpy as np

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
struct_dict = {
    (0, 0, 0, 0): '''
    [[0, 1, 1, 1, 1, 0],
     [0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 0],]
     ''',

    (0, 0, 0, 1): '''
    [[0, 1, 1, 1, 0, 0],
     [0, 0, 0, 0, 1, 0],
     [0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 0],]
     ''',

    (0, 0, 1, 1): '''
    [[0, 1, 1, 0, 0, 0],
     [0, 0, 0, 1, 1, 0],
     [0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 0],]
     ''',

    (0, 0, 1, 2): '''
    [[0, 1, 1, 0, 0, 0],
     [0, 0, 0, 1, 0, 0],
     [0, 0, 0, 0, 1, 0],
     [0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 0],]
     ''',
    (0, 0, 1, 3): '''
    [[0, 1, 1, 0, 0, 0],
     [0, 0, 0, 1, 0, 0],
     [0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 1, 0],
     [0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 0],]
     ''',
    (0, 1, 1, 1): '''
    [[0, 1, 0, 0, 0, 0],
     [0, 0, 1, 1, 1, 0],
     [0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 0],]
     ''',
    (0, 1, 1, 2): '''
    [[0, 1, 0, 0, 0, 0],
     [0, 0, 1, 1, 0, 0],
     [0, 0, 0, 0, 1, 0],
     [0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 0],]
     ''',
    (0, 1, 2, 2): '''
    [[0, 1, 0, 0, 0, 0],
     [0, 0, 1, 0, 0, 0],
     [0, 0, 0, 1, 1, 0],
     [0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 0],]
     ''',
    (0, 1, 2, 3): '''
    [[0, 1, 0, 0, 0, 0],
     [0, 0, 1, 0, 0, 0],
     [0, 0, 0, 1, 0, 0],
     [0, 0, 0, 0, 1, 0],
     [0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 0],]
    '''
    }

def exp_prompt_nasgraph(arch_dict):
    prompt1 = '''\nHere are some experimental results that you can use as a reference:\n'''  # 将 arch_list 和 acc_list 按照 acc_list 的元素大小进行排序
    prompt2 = '''\nThe model you propose should be strictly #different# from the structure of the existing experimental results.#You should not raise the models that are already present in the above experimental results again.#\n'''
    arch_l = list(arch_dict.keys())
    acc_l = [arch_dict[key] for key in arch_l]

    sorted_results = sorted(zip(arch_l, acc_l), key=lambda x: x[1], reverse=True)
    arch_l = [arch for arch, acc in sorted_results]
    acc_l = [acc for arch, acc in sorted_results]

    operation_repeat = set()
    operation_unique = []
    acc_unique = []
    seen = set()
    for i in range(len(arch_l)):
        if tuple(arch_l[i]) in seen:
            operation_repeat.add(arch_l[i])
        else:
            seen.add(tuple(arch_l[i]))
            operation_unique.append(arch_l[i])
            acc_unique.append(acc_l[i])

    prompt_repeat = ''''''
    if(len(operation_repeat)>0):
        prompt_repeat = '''In the above experimental results, there are some repetitive models, as follows\n{}. #Please do not make the mistake of presenting the existing model in the experimental results again!#\n'''.format(''.join(
        ['Model [{}]\n'.format(arch) for arch in operation_repeat]))

    prompt1 = prompt1 + '''{}#I hope you can learn the commonalities between the well performing models to achieve better results and avoid the mistakes of poor models to avoid achieving such poor results again.#\n''' \
        .format(''.join(
        ['Model [{}] achieves accuracy {:.4f} on the validation set.\n'.format(arch, acc) for arch, acc in
         zip(operation_unique, acc_unique)]))
    return prompt1 + prompt_repeat + prompt2

def prompt_nasgraph(dataname,link, arch_dict= None, stage=0):
    prompt1 = '''The task is to choose the best GNN  architecture on a given dataset. The architecture will be trained and tested on {}, and the objective is to maximize model accuracy.'''.format(
        dataname)
    link_prompt = '''
A GNN architecture is defined as follows: {
    The first operation is input, the last operation is output, and the intermediate operations are candidate operations.
    The adjacency matrix  of operation connections is as follows: ''' + struct_dict[tuple(link)] + '''where the (i,j)-th element in the adjacency matrix denotes that the output of operation $i$ will be used as  the input of operation $j$.
}'''
    operation_prompt='''
There are 9 operations that can be selected, including: gcn, gat, sage, gin, cheb, arma, graph, fc and skip.

The define for gat is as follows:
{
    The graph attentional operator from the "Graph Attention Networks" paper.
    $$\mathbf{x}_i^{\prime}=\\alpha_{i, i} \Theta \mathbf{x}_i+\sum_{j \in \mathcal{N}(i)} \\alpha_{i, j} \Theta \mathbf{x}_j$$
    where the attention coefficients $\\alpha_{i, j}$ are computed as
    $$\\alpha_{i, j}=\\frac{\exp \left(\operatorname{LeakyReLU}\left(\mathbf{a}^{\\top}\left[\\boldsymbol{\Theta} \mathbf{x}_i \| \Theta \mathbf{x}_j\\right]\\right)\\right)}{\sum_{k \in \mathcal{N}(i) \cup\{i\}} \exp \left(\operatorname{LeakyReLU}\left(\mathbf{a}^{\\top}\left[\Theta \mathbf{x}_i \| \Theta \mathbf{x}_k\\right]\\right)\\right)} .$$
}
The define for gcn is as follows:
{
    The graph convolutional operator from the "Semi-supervised Classification with Graph Convolutional Networks" paper.
    Its node-wise formulation is given by:
    $$\mathbf{x}_i^{\prime}=\\boldsymbol{\Theta}^{\\top} \sum_{j \in \mathcal{N}(i) \cup\{i\}} \\frac{e_{j, i}}{\sqrt{\hat{d}_j \hat{d}_i}} \mathbf{x}_j$$
    with $\hat{d}_i=1+\sum_{j \in \mathcal{N}(i)} e_{j, i}$, where $e_{j, i}$ denotes the edge weight from source node $j$ to target node i (default: 1.0 )
}
The define for gin is as follows:
{
    The graph isomorphism operator from the "How Powerful are Graph Neural Networks?" paper
    $$\mathbf{x}_i^{\prime}=h_{\Theta}\left((1+\epsilon) \cdot \mathbf{x}_i+\sum_{j \in \mathcal{N}(i)} \mathbf{x}_j\\right)$$
    or$$\mathbf{X}^{\prime}=h_{\Theta}((\mathbf{A}+(1+\epsilon) \cdot \mathbf{I}) \cdot \mathbf{X}),$$
    here $h_{\Theta}$ denotes a neural network, i.e. an MLP.
}
The define for cheb is as follows:
{
    The chebyshev spectral graph convolutional operator from the "Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering" paper
    $$\mathbf{X}^{\prime}=\sum_{k=1}^K \mathbf{Z}^{(k)} \cdot \Theta^{(k)}$$
    where $\mathbf{Z}^{(k)}$ is computed recursively by
    $$\\begin{aligned}
    & \mathbf{Z}^{(1)}=\mathbf{X} \\
    & \mathbf{Z}^{(2)}=\hat{\mathbf{L}} \cdot \mathbf{X} \\
    & \mathbf{Z}^{(k)}=2 \cdot \hat{\mathbf{L}} \cdot \mathbf{Z}^{(k-1)}-\mathbf{Z}^{(k-2)}
    \end{aligned}$$
    and $\hat{\mathbf{L}}$ denotes the scaled and normalized Laplacian $\\frac{2 \mathbf{L}}{\lambda_{\max }}-\mathbf{I}$.
}
The define for sage is as follows:
{
    The GraphSAGE operator from the "Inductive Representation Learning on Large Graphs" paper
    $$\mathbf{x}_i^{\prime}=\mathbf{W}_1 \mathbf{x}_i+\mathbf{W}_2 \cdot \operatorname{mean}_{j \in \mathcal{N}(i)} \mathbf{x}_j$$
}
The define for arma is as follows:
{
    The ARMA graph convolutional operator from the "Graph Neural Networks with Convolutional ARMAFilters" paper
    $$\mathbf{X}^{\prime}=\\frac{1}{K} \sum_{k=1}^K \mathbf{X}_k^{(T)}$$
    with $\mathbf{X}_k^{(T)}$ being recursively defined by
    $$\mathbf{X}_k^{(t+1)}=\sigma\left(\hat{\mathbf{L}} \mathbf{X}_k^{(t)} \mathbf{W}+\mathbf{X}^{(0)} \mathbf{V}\\right),$$
    where $\hat{\mathbf{L}}=\mathbf{I}-\mathbf{L}=\mathbf{D}^{-1 / 2} \mathbf{A} \mathbf{D}^{-1 / 2}$ denotes the modified Laplacian $\mathbf{L}=\mathbf{I}-\mathbf{D}^{-1 / 2} \mathbf{A} \mathbf{D}^{-1 / 2}$
}
The define for graph is as follows:
{
    The k-GNNs graph convolutional operator from the "Weisfeiler and Leman Go Neural: Higher-Order Graph Neural Networks" paper
    $$f_{k, \mathrm{~L}}^{(t)}(x)=\sigma\left(f_{k, \mathrm{~L}}^{(t-1)}(x) \cdot W_1^{(t)}+\sum_{u \in N_L(x)} f_{k, \mathrm{~L}}^{(t-1)}(u) \cdot W_2^{(t)}\\right) .$$
}
The define for fc layer is as follows:
{
$$\mathbf{x}^{\prime}=f(\Theta x+b)$$
}
The define for skip is as follows:
{
    $$\mathbf{x}^{\prime} = x$$
}
'''
    prompt2='''
Once again, your task is to help me find the optimal combination of operations corresponding to each layer of the model, while specifying the model structure and experimental dataset. The main difficulty of this task is how to reasonably arrange the selection strategy of the operation list, and each selected operation list corresponds to the highest accuracy that the operation can achieve. We should select new operation list to query based on the existing operation lists and their corresponding accuracy, in order to iteratively find the best operation list.

At the beginning, when there were few experimental results, we in the Exploration phase, we need to explore the operation space and identify which operation lists are promising. We can randomly select a batch of operation lists corresponding to each layer and evaluate their performance. Afterwards, we can sort the operation lists based on their accuracy and select some well performing operation lists as candidates for our Exploitation phase.

When we have a certain amount of experimental results, we are in the Exploitation phase, we focus on improving search by exploring the operating space more effectively. We can use optimization algorithms, such as Bayesian optimization or Evolutionary algorithm, to search for the best combination of operations, rather than randomly selecting the list of operations.
    '''

    notice1 = '''\n#Due to the lack of sufficient experimental results at present, it should be in the Exploration stage. You should focus more on exploring the entire search space evenly, rather than just focusing on the current local optimal results.#\n\n'''
    notice2 = '''\n#Due to the availability of a certain number of experimental results, I believe it is currently in the Exploitation stage. You should choose nearby samples that temporarily display the best results for searching, especially those that rank in the top 10% or 20% of existing experimental results. At the same time, you should try to avoid exploring sample structures with poor results, which can cause waste.#\n\n'''
    suffix = '''Please do not include anything other than the operation list in your response.
    And you should give 10 different models at a time, one model contains #4# operations.
    Your response only need include the operation list, for example: 
    1.Model: [arma,sage,graph,skip] 
    2.Model: [gat,fc,cheb,gin] 
    3. ...
    ......
    10.Model: [gcn,gcn,cheb,fc]. 
    And The response you give must strictly follow the format of this example. '''

    if(stage == 0):
        return prompt1+link_prompt+operation_prompt+prompt2+notice1+suffix
    elif(stage <4):
        return prompt1+link_prompt+operation_prompt+prompt2+ exp_prompt_nasgraph(arch_dict) + notice1 + suffix
    else:
        return prompt1+link_prompt+operation_prompt+prompt2+ exp_prompt_nasgraph(arch_dict) + notice2 + suffix


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


class nasgraph_SeachSpace(BaseSeachSpace):
    def __init__(self, dataname):
        link = best_link(dataname)
        candidate = {'operations': gnn_list,
                 'link': link}
        super().__init__(candidate)


    def eval(self,oper_str,data='cora'):
        operations =self.get_gnn(oper_str)
        link=self.candidate['link']
        bench = give_bench(data, link)
        arch = Arch(link, operations)
        info = bench[arch.valid_hash()]
        ans = {'valid_perf': info['valid_perf'],
               'test_perf': info['perf'],
               'bench': info}
        return ans


    def get_gnn(self,oper_str):

        operations_list = oper_str.split(',')
        return operations_list

    def prompt_ger(self, data_name,stage=0):
        if stage == 0:
            return prompt_nasgraph(data_name, self.candidate['link'],stage=stage)
        arch_dict={}
        for key in self.arch_dict.keys():
            arch_dict[key]=self.arch_dict[key]['valid_perf']
        return prompt_nasgraph(self.dataname,self.candidate['link'],arch_dict=arch_dict,stage=stage)

class nasgraph_trainer(BaseTrainer):
    def __init__(self, SeachSpace, LLM, file_address='./history/GNAS_LLM/', dataname='cora'):
        super().__init__(SeachSpace, LLM, file_address)
        self.dataname=dataname

    def test(self,dataname):
        result=self.SeachSpace.eval(self.arch,dataname)
        return result

    def res_to_str(self,res):

        input_lst = res.split('Model:')

        ans = []
        for i in range(1, len(input_lst)):
            operations_str = input_lst[i].split('[')[1].split(']')[0]
            operations_list_str = operations_str.replace(" ", "")
            if operations_list_str == ['skip,skip,skip,skip']:
                continue
            ans.append(operations_list_str)
        return ans

