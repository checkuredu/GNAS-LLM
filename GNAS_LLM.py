from nas_bench_graph import light_read as lightread
from nas_bench_graph import Arch
from untils import best_link, give_bench, main_prompt
from base import Basetrainer,BaseSeachSpace
import json
import requests
import time
import numpy as np
#url = "https://api-hk.openai-sb.com/v1/chat/completions"
url = "https://api.openai-sb.com/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer " + "sb-0538b4c5b057d61a6291b36877025b9150121dbff8c8be51"
}
system_content = '''Please pay special attention to my use of special markup symbols in the content below.The special markup symbols is # # ,and the content that needs special attention will be between #.'''
dataname = 'cora'
link = best_link(dataname)
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

operation_dict ={
    'gat':'''
    The graph attentional operation from the "Graph Attention Networks" paper.
    $$\mathbf{x}_i^{\prime}=\\alpha_{i, i} \Theta \mathbf{x}_i+\sum_{j \in \mathcal{N}(i)} \\alpha_{i, j} \Theta \mathbf{x}_j$$
    where the attention coefficients $\\alpha_{i, j}$ are computed as
    $$\\alpha_{i, j}= \\frac{\exp \left({LeakyReLU}\left(\mathbf{a}^{\\top}\left[\\boldsymbol{\Theta} \mathbf{x}_i | \Theta \mathbf{x}_j\\right]\\right)\\right)}{\sum_{k \in \mathcal{N}(i) \cup\{i\}} \exp \left({LeakyReLU}\left(\mathbf{a}^{\\top}\left[\Theta \mathbf{x}_i | \Theta \mathbf{x}_k\\right]\\right)\\right)} $$''',
    'gcn':'''
    The graph convolutional operation from the "Semi-supervised Classification with Graph Convolutional Networks" paper.
    Its node-wise formulation is given by:
    $$\mathbf{x}_i^{\prime}=\\boldsymbol{\Theta}^{\\top} \sum_{j \in \mathcal{N}(i) \cup\{i\}} \\frac{e_{j, i}}{\sqrt{\hat{d}_j \hat{d}_i}} \mathbf{x}_j$$
    with$\hat{d}_i=1+\sum_{j \in \mathcal{N}(i)} e_{j, i}$
    , where $e_{j, i}$ denotes the edge weight from source node $j$ to target node i (default: 1.0 )''',
    'sage':'''
    The GraphSAGE operation from the "Inductive Representation Learning on Large Graphs" paper
    $$\mathbf{x}_i^{\prime}=\Theta_1 \mathbf{x}_i+\Theta_2 \cdot {mean}_{j \in \mathcal{N}(i)} \mathbf{x}_j$$''',
    'gin':'''
    The graph isomorphism operation from the "How Powerful are Graph Neural Networks?" paper
    $$\mathbf{x}_i^{\prime}=h_{\Theta}\left((1+\epsilon) \cdot \mathbf{x}_i+\sum_{j \in \mathcal{N}(i)} \mathbf{x}_j\\right)$$
    here $h_{\Theta}$ denotes a neural network, i.e. an MLP.''',
    'cheb':'''
    The chebyshev spectral graph convolutional operation from the "Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering" paper
    $$\mathbf{X}^{\prime}=\sum_{k=1}^2 \mathbf{Z}^{(k)} \cdot \Theta^{(k)}$$
    where $\mathbf{Z}^{(k)}$ is computed recursively by\quad
    $\mathbf{Z}^{(1)}=\mathbf{X} \quad \mathbf{Z}^{(2)}=\hat{\mathbf{L}} \cdot \mathbf{X}$ and $\hat{\mathbf{L}}$\quad denotes the scaled and normalized Laplacian $\\frac{2 \mathbf{L}}{\lambda_{\max }}-\mathbf{I}$.''',
    'arma':'''
    The ARMA graph convolutional operation from the "Graph Neural Networks with Convolutional ARMAFilters" paper
    $$\mathbf{X}^{\prime}= \mathbf{X}_1^{(1)}$$
    with $\mathbf{X}_1^{(1)}$ being recursively defined by
    $\mathbf{X}_1^{(1)}=\sigma\left(\hat{\mathbf{L}} \mathbf{X}_1^{(0)} \Theta+\mathbf{X}^{(0)} \mathbf{V}\\right)$
    where $\hat{\mathbf{L}}=\mathbf{I}-\mathbf{L}=\mathbf{D}^{-1 / 2} \mathbf{A} \mathbf{D}^{-1 / 2}$ denotes the modified Laplacian $\mathbf{L}=\mathbf{I}-\mathbf{D}^{-1 / 2} \mathbf{A} \mathbf{D}^{-1 / 2}$''',
    'graph':'''
    The k-GNNs graph convolutional operation from the "Weisfeiler and Leman Go Neural: Higher-Order Graph Neural Networks" paper
    $$\mathbf{x}_i^{\prime}=\Theta_1\mathbf{x}_i+\Theta_2\sum_{j\in\mathcal{N}(i)}e_{j,i}\cdot\mathbf{x}_j$$''',
    'skip':'''
    $$\mathbf{x}^{\prime} = x$$''',
    'fc':'''
    $$\mathbf{x}^{\prime}=f(\Theta x+b)$$'''}


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
notice1 = '''\n#Due to the lack of sufficient experimental results at present, it should be in the Exploration stage. You should focus more on exploring the entire search space evenly, rather than just focusing on the current local optimal results.#\n\n'''
notice2 = '''\n#Due to the availability of a certain number of experimental results, I believe it is currently in the Exploitation stage. You should choose nearby samples that temporarily display the best results for searching, especially those that rank in the top 10% or 20% of existing experimental results. At the same time, you should try to avoid exploring sample structures with poor results, which can cause waste.#\n\n'''
link = best_link(dataname)
link_prompt = '''A GNN architecture is defined as follows: {
    The first operation is input, the last operation is output, and the intermediate operations are candidate operations.
    The adjacency matrix  of operation connections is as follows: '''+struct_dict[tuple(link)]+'''where the (i,j)-th element in the adjacency matrix denotes that the output of operation $i$ will be used as  the input of operation $j$.
}'''
strategy = [notice1,notice2]
suffix = '''Please do not include anything other than the operation list in your response.
    And you should give 10 different models at a time, one model contains #4# operations.
    Your response only need include the operation list, for example: 
    1.Model: [arma,sage,graph,skip] 
    2.Model: [gat,fc,cheb,gin] 
    3. ...
    ......
    10.Model: [gcn,gcn,cheb,fc]. 
    And The response you give must strictly follow the format of this example. '''
bench = give_bench(dataname,link)
def LLM(prompt,dataname):
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt},
    ]
    da = {
        "model": "gpt-3.5-turbo",  # "gpt-4"-0314
        "messages": messages,
        "temperature": 0
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(da))
        response.raise_for_status()  # 检查请求是否成功
        res = response.json()
    except (requests.HTTPError, json.JSONDecodeError) as e:
        print("JSON解析错误:", e)
    except Exception as e:
        print("其他异常:", e)

    res_temp = res['choices'][0]['message']['content']
    # print(res_temp)
    input_lst = res_temp.split('Model:')

    ans = []
    for i in range(1, len(input_lst)):
        operations_str = input_lst[i].split('[')[1].split(']')[0]
        operations_list = operations_str.split(',')
        operations_list_str = [a.replace(" ", "") for a in operations_list]
        if operations_list_str == ['skip', 'skip', 'skip', 'skip']:
            continue
        ans.append(operations_list_str)
    return ans

def eval(operations,dataname):
    link = best_link(dataname)
    bench = give_bench(dataname, link)
    arch = Arch(link, operations)
    info = bench[arch.valid_hash()]
    return info['perf']

GNASLLMSS = BaseSeachSpace(suffix=suffix,evaluate=eval, operation_dict=operation_dict,link_prompt=link_prompt,strategy=strategy)
GNASLLMtrainer = Basetrainer(GNASLLMSS, LLM)

best_arch = GNASLLMtrainer.fit('cora')
print(best_arch)