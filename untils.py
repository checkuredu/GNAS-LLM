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

def experiments_prompt(arch_list, acc_list, dataname):
    prompt1 = '''\nHere are some experimental results that you can use as a reference:\n'''  # 将 arch_list 和 acc_list 按照 acc_list 的元素大小进行排序
    prompt2 = '''\nPlease propose 10 better and #different# models with accuracy strictly greater than {}, which can improve the performance of the model on {} in addition to the experimental results mentioned above.\n'''.format(
        sum(acc_list) / len(acc_list), dataname)
    prompt3 = '''\nThe model you propose should be strictly #different# from the structure of the existing experimental results.#You should not raise the models that are already present in the above experimental results again.#\n'''

    operation_repeat = []
    operation_unique = []
    acc_unique = []
    seen = set()
    for i in range(len(arch_list)):
        if tuple(arch_list[i]['arch_Operations']) in seen:
            operation_repeat.append(arch_list[i]['arch_Operations'])
        else:
            seen.add(tuple(arch_list[i]['arch_Operations']))
            operation_unique.append(arch_list[i]['arch_Operations'])
            acc_unique.append(acc_list[i])
    prompt_repeat = ''''''
    if(len(operation_repeat)>0):
        prompt_repeat = '''In the above experimental results, there are some repetitive models, as follows\n{}. #Please do not make the mistake of presenting the existing model in the experimental results again!#\n'''.format(''.join(
        ['Model [{}]\n'.format(arch) for arch in operation_repeat]))

    prompt1 = prompt1 + '''{}#I hope you can learn the commonalities between the well performing models to achieve better results and avoid the mistakes of poor models to avoid achieving such poor results again.#\n''' \
        .format(''.join(
        ['Model [{}] achieves accuracy {:.4f} on the validation set.\n'.format(arch['arch_Operations'], acc) for arch, acc in
         zip(operation_unique, acc_unique)]))
    return prompt1 + prompt_repeat + prompt2 + prompt3

def main_prompt(link, dataname, arch_list = None, acc_list = None, stage=0):
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

    user_input = '''The task is to choose the best GNN  architecture on a given dataset. The architecture will be trained and tested on '''+dataname+''',  and the objective is to maximize model accuracy.
A GNN architecture is defined as follows: 
{
    The first operation is input, the last operation is output, and the intermediate operations are candidate operations.
    The adjacency matrix  of operation connections is as follows: '''+struct_dict[link]+'''where the (i,j)-th element in the adjacency matrix denotes that the output of operation $i$ will be used as  the input of operation $j$.
}

There are 9 operations that can be selected, including 7 most widely adopted GNN operations: GCN, GAT, GraphSAGE, GIN, ChebNet, ARMA, k-GNN, skip connection, and fully connected layer.
The definition of GAT is as follows:
{
    The graph attentional operation from the "Graph Attention Networks" paper.
    $$\mathbf{x}_i^{\prime}=\\alpha_{i, i} \Theta \mathbf{x}_i+\sum_{j \in \mathcal{N}(i)} \\alpha_{i, j} \Theta \mathbf{x}_j$$
    where the attention coefficients $\\alpha_{i, j}$ are computed as
    $$\\alpha_{i, j}= \\frac{\exp \left({LeakyReLU}\left(\mathbf{a}^{\\top}\left[\\boldsymbol{\Theta} \mathbf{x}_i | \Theta \mathbf{x}_j\\right]\\right)\\right)}{\sum_{k \in \mathcal{N}(i) \cup\{i\}} \exp \left({LeakyReLU}\left(\mathbf{a}^{\\top}\left[\Theta \mathbf{x}_i | \Theta \mathbf{x}_k\\right]\\right)\\right)} $$
}
The definition of GCN is as follows:
{
    The graph convolutional operation from the "Semi-supervised Classification with Graph Convolutional Networks" paper.
    Its node-wise formulation is given by:
    $$\mathbf{x}_i^{\prime}=\\boldsymbol{\Theta}^{\\top} \sum_{j \in \mathcal{N}(i) \cup\{i\}} \\frac{e_{j, i}}{\sqrt{\hat{d}_j \hat{d}_i}} \mathbf{x}_j$$
    with$\hat{d}_i=1+\sum_{j \in \mathcal{N}(i)} e_{j, i}$
    , where $e_{j, i}$ denotes the edge weight from source node $j$ to target node i (default: 1.0 )
}
The definition of GIN is as follows:
{
    The graph isomorphism operation from the "How Powerful are Graph Neural Networks?" paper
    $$\mathbf{x}_i^{\prime}=h_{\Theta}\left((1+\epsilon) \cdot \mathbf{x}_i+\sum_{j \in \mathcal{N}(i)} \mathbf{x}_j\\right)$$
    here $h_{\Theta}$ denotes a neural network, i.e. an MLP.
}
The definition of ChebNet is as follows:
{
    The chebyshev spectral graph convolutional operation from the "Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering" paper
    $$\mathbf{X}^{\prime}=\sum_{k=1}^2 \mathbf{Z}^{(k)} \cdot \Theta^{(k)}$$
                where $\mathbf{Z}^{(k)}$ is computed recursively by\quad
                $\mathbf{Z}^{(1)}=\mathbf{X} \quad \mathbf{Z}^{(2)}=\hat{\mathbf{L}} \cdot \mathbf{X}$
                \quad and\quad $\hat{\mathbf{L}}$\quad denotes the scaled and normalized Laplacian $\\frac{2 \mathbf{L}}{\lambda_{\max }}-\mathbf{I}$.
}
The definition of GraphSAGE is as follows:
{
    The GraphSAGE operation from the "Inductive Representation Learning on Large Graphs" paper
    $$\mathbf{x}_i^{\prime}=\Theta_1 \mathbf{x}_i+\Theta_2 \cdot {mean}_{j \in \mathcal{N}(i)} \mathbf{x}_j$$
}
The definition of ARMA is as follows:
{
    The ARMA graph convolutional operation from the "Graph Neural Networks with Convolutional ARMAFilters" paper
    $$\mathbf{X}^{\prime}= \mathbf{X}_1^{(1)}$$
    with $\mathbf{X}_1^{(1)}$ being recursively defined by
    $\mathbf{X}_1^{(1)}=\sigma\left(\hat{\mathbf{L}} \mathbf{X}_1^{(0)} \Theta+\mathbf{X}^{(0)} \mathbf{V}\\right)$
    where $\hat{\mathbf{L}}=\mathbf{I}-\mathbf{L}=\mathbf{D}^{-1 / 2} \mathbf{A} \mathbf{D}^{-1 / 2}$ denotes the modified Laplacian $\mathbf{L}=\mathbf{I}-\mathbf{D}^{-1 / 2} \mathbf{A} \mathbf{D}^{-1 / 2}$
}
The definition of k-GNN is as follows:
{
    The k-GNNs graph convolutional operation from the "Weisfeiler and Leman Go Neural: Higher-Order Graph Neural Networks" paper
    $$\mathbf{x}_i^{\prime}=\Theta_1\mathbf{x}_i+\Theta_2\sum_{j\in\mathcal{N}(i)}e_{j,i}\cdot\mathbf{x}_j$$
}
The definition of the fully connected layer is as follows:
{
    $$\mathbf{x}^{\prime}=f(\Theta x+b)$$
}
The definition of skip connection is as follows:
{
    $$\mathbf{x}^{\prime} = x$$
}

Once again, your task is to help me find the optimal combination of operations while specifying the GNN architecture and experimental dataset. The main difficulty of this task is how to reasonably arrange the selection strategy of the operation list, and each selected operation list corresponds to the highest accuracy that the operation can achieve. We should select a new operation list to query based on the existing operation lists and their corresponding accuracy, in order to iteratively find the best operation list.

At the beginning, when there were few experimental results, we in the Exploration phase, we need to explore the operation space and identify which operation lists are promising. We can randomly select a batch of operation lists corresponding to each layer and evaluate their performance. Afterwards, we can sort the operation lists based on their accuracy and select some well performing operation lists as candidates for our Exploitation phase.

When we have a certain amount of experimental results, we are in the Exploitation phase, we focus on improving search by exploring the operating space more effectively. We can use optimization algorithms, such as Bayesian optimization or Evolutionary algorithm, to search for the best combination of operations, rather than randomly selecting the list of operations.
    
'''
    notice1 = '''\n#Due to the lack of sufficient experimental results at present, it should be in the Exploration stage. You should focus more on exploring the entire search space evenly, rather than just focusing on the current local optimal results.#\n\n'''
    notice2 = '''\n#Due to the availability of a certain number of experimental results, I believe it is currently in the Exploitation stage. You should choose nearby samples that temporarily display the best results for searching, especially those that rank in the top 10% or 20% of existing experimental results. At the same time, you should try to avoid exploring sample structures with poor results, which can cause waste.#\n\n'''
    suffix = '''Please do not include anything other than the operation list in your response.
    And you should give 10 different models at a time, one model contains #4# operations.
    Your response only need include the operation list, for example: 
    1.model: [ARMA,GraphSAGE,k-GNN,skip] 
    2.model: [GAT,fully-connected-layer,ChebNet,GIN] 
    3. ...
    ......
    10.model: [GCN,GCN,ChebNet,fully-connected-layer]. 
    And The response you give must strictly follow the format of this example. '''

    if(stage == 0):
        return user_input+notice1+suffix
    elif(stage < 4):
        return user_input + experiments_prompt(arch_list, acc_list, dataname) + notice1 + suffix
    else:
        return user_input + experiments_prompt(arch_list, acc_list, dataname) + notice2 + suffix
