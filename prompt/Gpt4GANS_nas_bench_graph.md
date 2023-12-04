# Prompt for NAS-Bench-Graph

## System Prompt
```
Please pay special attention to my use of special markup symbols in the content below.
The special markup symbols is # # ,and the content that needs special attention will be between #.
```
## main prompt
```
The task is to choose the best GNN  architecture on a given dataset. The architecture will be trained and tested on  "dataname",  and the objective is to maximize model accuracy.
A GNN architecture is defined as follows: 
{
    The first operation is input, the last operation is output, and the intermediate operations are candidate operations.
    The adjacency matrix  of operation connections is as follows: 
    [sturct_array]
    where the (i,j)-th element in the adjacency matrix denotes that the output of operation $i$ will be used as  the input of operation $j$.
}

There are 9 operations that can be selected, including 7 most widely adopted GNN operations: GCN, GAT, GraphSAGE, GIN, ChebNet, ARMA, k-GNN, skip connection, and fully connected layer. 
The definition of GAT is as follows:
{
    The graph attentional operation from the "Graph Attention Networks" paper.
    $$\mathbf{x}_i^{\prime}=\alpha_{i, i} \Theta \mathbf{x}_i+\sum_{j \in \mathcal{N}(i)} \alpha_{i, j} \Theta \mathbf{x}_j$$
    where the attention coefficients $\alpha_{i, j}$ are computed as
    $$\alpha_{i, j}= \frac{\exp \left({LeakyReLU}\left(\mathbf{a}^{\top}\left[\boldsymbol{\Theta} \mathbf{x}_i | \Theta \mathbf{x}_j\right]\right)\right)}{\sum_{k \in \mathcal{N}(i) \cup\{i\}} \exp \left({LeakyReLU}\left(\mathbf{a}^{\top}\left[\Theta \mathbf{x}_i | \Theta \mathbf{x}_k\right]\right)\right)} $$
}
The definition of GCN is as follows:
{
    The graph convolutional operation from the "Semi-supervised Classification with Graph Convolutional Networks" paper.
    Its node-wise formulation is given by:
    $$\mathbf{x}_i^{\prime}=\boldsymbol{\Theta}^{\top} \sum_{j \in \mathcal{N}(i) \cup\{i\}} \frac{e_{j, i}}{\sqrt{\hat{d}_j \hat{d}_i}} \mathbf{x}_j$$
    with$\hat{d}_i=1+\sum_{j \in \mathcal{N}(i)} e_{j, i}$
    , where $e_{j, i}$ denotes the edge weight from source node $j$ to target node i (default: 1.0 )
} 
The definition of GIN is as follows:
{
    The graph isomorphism operation from the "How Powerful are Graph Neural Networks?" paper
    $$\mathbf{x}_i^{\prime}=h_{\Theta}\left((1+\epsilon) \cdot \mathbf{x}_i+\sum_{j \in \mathcal{N}(i)} \mathbf{x}_j\right)$$
    here $h_{\Theta}$ denotes a neural network, i.e. an MLP.
}
The definition of ChebNet is as follows:
{
    The chebyshev spectral graph convolutional operation from the "Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering" paper
    $$\mathbf{X}^{\prime}=\sum_{k=1}^2 \mathbf{Z}^{(k)} \cdot \Theta^{(k)}$$
                where $\mathbf{Z}^{(k)}$ is computed recursively by\quad
                $\mathbf{Z}^{(1)}=\mathbf{X} \quad \mathbf{Z}^{(2)}=\hat{\mathbf{L}} \cdot \mathbf{X}$
                \quad and\quad $\hat{\mathbf{L}}$\quad denotes the scaled and normalized Laplacian $\frac{2 \mathbf{L}}{\lambda_{\max }}-\mathbf{I}$.
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
    $\mathbf{X}_1^{(1)}=\sigma\left(\hat{\mathbf{L}} \mathbf{X}_1^{(0)} \Theta+\mathbf{X}^{(0)} \mathbf{V}\right)$
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

At the beginning, when only a few numbers of evaluated architectures are available, use the exploration strategy to explore the operations.  Randomly select a batch of operations for evaluation

When a certain amount of evaluated architectures are available, use the exploitation strategy to find the best operations by sampling the best candidate operations from previously generated candidates.

#Due to the lack of sufficient experimental results at present, it should be in the Exploration stage. You should focus more on exploring the entire search space evenly, rather than just focusing on the current local optimal results.#

Please do not include anything other than the operation list in your response.
And you should give 10 different model operation lists at once.
Your response only need to include the operation list, for example: 
1.model: [arma,sage,graph,skip]
2.model: [gat,fc,cheb,gin]
3. ...
...
10.model: [gcn,gcn,cheb,fc].
And The response you give must strictly follow the format of this example. 
```