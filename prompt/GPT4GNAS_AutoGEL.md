# Prompt for NAS-Bench-Graph

## System Prompt
```
Please pay special attention to my use of special markup symbols in the content below.
The special markup symbols is # # ,and the content that needs special attention will be between #.
```
## main prompt
```
The task is to choose the best GNN  architecture on a given dataset. The architecture will be trained and tested on  "dataname",  and the objective is to maximize model accuracy.
The definition of search space for models follow the neighborhood aggregation schema, i.e., the Message Passing Neural Network (MPNN), which is formulated as:
{
    $$\mathbf{m}_{v}^{k+1}=AGG_{k}(\{M_{k}(\mathbf{h}_{v}^{k}\mathbf{h}_{u}^{k},\mathbf{e}_{vu}^{k}):u\in N(v)\})$$
    $$\mathbf{h}_{v}^{k+1}=ACT_{k}(COM_{k}(\{\mathbf{h}_{v}^{k},\mathbf{m}_{v}^{k+1}\}))$$
    $$\hat{\mathbf{y}}=R(\{\mathbf{h}_{v}^{L}|v\in G\})$$
where $k$ denotes $k$-th layer, $N(v)$ denotes a set of neighboring nodes of $v$, $\mathbf{h}_{v}^{k}$, $\mathbf{h}_{u}^{k}$ denotes hidden embeddings for $v$ and $u$ respectively, $\mathrm{e}_{vu}^{k}$ denotes features for edge e(v, u) (optional), $\mathbf{m}_{v}^{k+1}$denotes the intermediate embeddings gathered from neighborhood $N(v)$, $M_k$ denotes the message function, $AGG_{k}$ denotes the neighborhood aggregation function, $COM_{k}$ denotes the combination function between intermediate embeddings and embeddings of node $v$ itself from the last layer, $ACT_{k}$ denotes activation function. Such message passing phase in repeats for $L$ times (i.e.,$ k\in\{1,\cdots,L\}$). For graph-level tasks, it further follows the readout phase in where information from the entire graph $G$ is aggregated through readout function $R(·)$.
}
The parts that need to be determined in the search space through architecture search, along with their corresponding selectable functions, are as follows:
{
    agg: [sum,mean,max];
    combine: [sum,concat];
    act: [relu,prelu];
    layer_connect: [stack,skip_sum,skip_cat];
    layer_agg: [none,concat,max_pooling];
}
#Define the model as a two-layer GNN model, where you need to choose functions agg, combine, act, and layer_connect for each layer, and also need to select a layer_agg function for the entire model.#

Once again, your task is to help me find the optimal model on the dataset of "dataname". The main difficulty of this task lies in how to reasonably arrange the search strategies in different stages. We should choose a new model to try based on the existing model and its corresponding accuracy, so as to iteratively find the best model.

At the beginning, when there were few experimental results, we in the Exploration phase, we need to explore the search space and identify which model is promising. We can randomly select a batch of models and evaluate their performance. Afterwards, we can sort the models based on their accuracy and select some well performing models as candidates for our Exploitation phase.

When we have a certain amount of experimental results, we are in the Exploitation phase, we focus on improving search by exploring the search space more effectively. We can use optimization algorithms, such as Bayesian optimization or evolutionary algorithms, to focus on the vicinity of those models that perform best among the models that have been searched for experimental results, rather than randomly selecting them.
```