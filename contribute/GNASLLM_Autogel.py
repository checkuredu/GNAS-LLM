from base.base import BaseSeachSpace, BaseTrainer
import json
import requests
import argparse
import torch
from torch_geometric.data import DataLoader
from AutoGEL.AutoGEL_main import main_autogel,autogel_getmodel,retrain
from AutoGEL.utils import reset_mask,get_loader
from AutoGEL.train import eval_model
import time
import numpy as np


def exp_prompt_autogel(arch_list, acc_list):
    prompt1 = '''Here are some experimental results that you can use as a reference:\n'''
    # 将 arch_list 和 acc_list 按照 acc_list 的元素大小进行排序
    sorted_results = sorted(zip(arch_list, acc_list), key=lambda x: x[1], reverse=True)
    better = '''{}#I hope you can learn the commonalities between the well performing models to achieve better results and avoid the mistakes of poor models to avoid achieving such poor results again.#\n'''\
        .format(''.join(['The model {} gives an accuracy of {:.3f}\n'.format(arch, acc) for arch, acc in sorted_results]))

    prompt2 = '''\nPlease propose 10 better and #different# models, which can improve the performance in addition to the experimental results mentioned above.\n'''
    prompt3 = '''\nThe models you propose should be strictly #different# from the existing experimental results.#You should not raise the model that are already present in the above experimental results again.#\n'''

    return prompt1 + better + prompt2 + prompt3
def prompt_autogel(dataname,arch_list=None,acc_list=None,stage=0):
    prompt1 = '''The task is to choose the best GNN  architecture on a given dataset. The architecture will be trained and tested on {}, and the objective is to maximize model accuracy.'''.format(
        dataname)

    link_prompt = '''
The definition of search space for models follow the neighborhood aggregation schema, i.e., the Message Passing Neural Network (MPNN), which is formulated as:
{
    $$\mathbf{m}_{v}^{k+1}=AGG_{k}(\{M_{k}(\mathbf{h}_{v}^{k}\mathbf{h}_{u}^{k},\mathbf{e}_{vu}^{k}):u\in N(v)\})$$
    $$\mathbf{h}_{v}^{k+1}=ACT_{k}(COM_{k}(\{\mathbf{h}_{v}^{k},\mathbf{m}_{v}^{k+1}\}))$$
    $$\hat{\mathbf{y}}=R(\{\mathbf{h}_{v}^{L}|v\in G\})$$
    where $k$ denotes $k$-th layer, $N(v)$ denotes a set of neighboring nodes of $v$, $\mathbf{h}_{v}^{k}$, $\mathbf{h}_{u}^{k}$ denotes hidden embeddings for $v$ and $u$ respectively, $\mathrm{e}_{vu}^{k}$ denotes features for edge e(v, u) (optional), $\mathbf{m}_{v}^{k+1}$denotes the intermediate embeddings gathered from neighborhood $N(v)$, $M_k$ denotes the message function, $AGG_{k}$ denotes the neighborhood aggregation function, $COM_{k}$ denotes the combination function between intermediate embeddings and embeddings of node $v$ itself from the last layer, $ACT_{k}$ denotes activation function. Such message passing phase in repeats for $L$ times (i.e.,$ k\in\{1,\cdots,L\}$). For graph-level tasks, it further follows the readout phase in where information from the entire graph $G$ is aggregated through readout function $R(·)$.
}'''
    operation_prompt='''
The parts that need to be determined in the search space through architecture search, along with their corresponding selectable functions, are as follows:
{
    agg: [sum,mean,max];
    combine: [sum,concat];
    act: [relu,prelu];
    layer_connect: [stack,skip_sum,skip_cat];
    layer_agg: [none,concat,max_pooling];
}
#Define the model as a two-layer GNN model, where you need to choose functions agg, combine, act, and layer_connect for each layer, and also need to select a layer_agg function for the entire model.#
'''
    prompt2='''
Once again, your task is to help me find the optimal combination of operations corresponding to each layer of the model, while specifying the model structure and experimental dataset. The main difficulty of this task is how to reasonably arrange the selection strategy of the operation list, and each selected operation list corresponds to the highest accuracy that the operation can achieve. We should select new operation list to query based on the existing operation lists and their corresponding accuracy, in order to iteratively find the best operation list.

At the beginning, when there were few experimental results, we in the Exploration phase, we need to explore the operation space and identify which operation lists are promising. We can randomly select a batch of operation lists corresponding to each layer and evaluate their performance. Afterwards, we can sort the operation lists based on their accuracy and select some well performing operation lists as candidates for our Exploitation phase.

When we have a certain amount of experimental results, we are in the Exploitation phase, we focus on improving search by exploring the operating space more effectively. We can use optimization algorithms, such as Bayesian optimization or Evolutionary algorithm, to search for the best combination of operations, rather than randomly selecting the list of operations.
    '''

    notice1 = '''\n#Due to the lack of sufficient experimental results at present, it should be in the Exploration stage. You should focus more on exploring the entire search space evenly, rather than just focusing on the current local optimal results.#\n\n'''
    notice2 = '''\n#Due to the availability of a certain number of experimental results, I believe it is currently in the Exploitation stage. You should choose nearby samples that temporarily display the best results for searching, especially those that rank in the top 10% or 20% of existing experimental results. At the same time, you should try to avoid exploring sample structures with poor results, which can cause waste.#\n\n'''
    suffix = '''Please answer me as briefly as possible. You should give 10 different model structures at a time.
    For example:
    1.Model:{layer1:{ agg:sum, combine:concat, act:prelu, layer_connect:skip_cat}; layer2:{ agg:sum, combine:concat, act:relu,layer_connect:stack}; layer_agg:concat}
    2.Model:{layer1:{ agg:mean, combine:sum, act:relu, layer_connect:skip_sum}; layer2:{ agg:max, combine:sum, act:prelu, layer_connect:skip_cat}; layer_agg:max_pooling}
    3.Model: ...
    ...
    10.Model:{layer1:{ agg:max, combine:sum, act:relu, layer_connect:stack}; layer2:{agg:mean, combine:sum, act:relu, layer_connect:skip_sum}; layer_agg:concat}
    And The response you give must strictly follow the format of this example. '''

    if(stage == 0):
        return prompt1+link_prompt+operation_prompt+prompt2+notice1+suffix
    elif(stage <4):
        return prompt1+link_prompt+operation_prompt+prompt2+ exp_prompt_autogel(arch_list, acc_list) + notice1 + suffix
    else:
        return prompt1+link_prompt+operation_prompt+prompt2+ exp_prompt_autogel(arch_list, acc_list) + notice2 + suffix

class autogel_SeachSpace(BaseSeachSpace):
    def __init__(self):
        candidate={ 'agg': ['sum', 'mean', 'max'],
                    'combine': ['sum', 'concat'],
                    'act': ['relu', 'prelu'],
                    'layer_connect': ['stack', 'skip_sum', 'skip_cat'],
                    'layer_agg': ['none', 'concat', 'max_pooling'],
                    'pool': ['global_add_pool', 'global_mean_pool','global_max_pool']}
        super().__init__(candidate)

    # def eval(self, arch, data):
    #     model_str_list = arch.split(';')
    #     layer_str = {'agg': ['', ''],
    #                  'combine': ['', ''],
    #                  'act': ['', ''],
    #                  'layer_connect': ['', ''],
    #                  'layer_agg': ['']}
    #     layer_str['layer_agg'][0] = model_str_list[2].split('layer_agg:')[1]
    #     for j in range(2):
    #         layer_str['agg'][j] = model_str_list[j].split('agg:')[1].split(',')[0]
    #         layer_str['combine'][j] = model_str_list[j].split('combine:')[1].split(',')[0]
    #         layer_str['act'][j] = model_str_list[j].split('act:')[1].split(',')[0]
    #         layer_str['layer_connect'][j] = model_str_list[j].split('layer_connect:')[1].split('}')[0]
    #     results=main_autogel(layer_str, self.dataname, seed=10, gpu=0)
    #     return results

    def eval(self,arch,data):
        in_features = max(data.num_node_features, 1)
        out_features = data.num_classes
        model=self.get_gnn(arch, in_features, out_features)

        parser = argparse.ArgumentParser('Interface for evaluate autoGEL')
        parser.add_argument('--task', type=str, default='node', help='type of task', choices=['node', 'graph'])
        parser.add_argument('--dataset', type=str, default='Cora',
                            help='dataset name')  # choices=['PROTEINS', 'IMDB-BINARY', 'IMDB-MULTI', 'BZR', 'COX2', 'DD', 'ENZYMES', 'NCI1']
        parser.add_argument('--train_ratio', type=float, default=0.6, help='ratio of the train against whole')
        parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio of the val against whole')
        parser.add_argument('--test_ratio', type=float, default=0.2, help='ratio of the test against whole')
        parser.add_argument('--bs', type=int, default=128, help='minibatch size')
        parser.add_argument('--gpu', type=int, default=0, help='gpu id')
        parser.add_argument('--optimizer', type=str, default='adam', help='optimizer to use')
        parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
        parser.add_argument('--l2', type=float, default=1e-4, help='l2 regularization weight')
        parser.add_argument('--metric', type=str, default='acc', help='metric for evaluating performance',
                            choices=['acc', 'auc'])
        parser.add_argument('--retrain_epoch', type=int, default=200, help='number of epochs to retrain')
        parser.add_argument('--clip', type=float, default=0.1, help='gradient clipping')
        args = parser.parse_args()

        train_mask, val_mask, test_mask = reset_mask(data, args=args, stratify=None)
        train_loader, val_loader, test_loader = get_loader(data, args=args)
        model, results = retrain(model, train_loader, val_loader, test_loader, train_mask, val_mask, test_mask, args)
        return results

    def get_gnn(self,oper_str,in_features, out_features,gpu=0):
        op2z_mapping = {
            'agg': {'sum': [1.0, 0.0, 0.0], 'mean': [0.0, 1.0, 0.0], 'max': [0.0, 0.0, 1.0]},
            'combine': {'sum': [1.0, 0.0], 'concat': [0.0, 1.0]},
            'act': {'relu': [1.0, 0.0], 'prelu': [0.0, 1.0]},
            'layer_connect': {'stack': [1.0, 0.0, 0.0], 'skip_sum': [0.0, 1.0, 0.0], 'skip_cat': [0.0, 0.0, 1.0]},
            'layer_agg': {'none': [1.0, 0.0, 0.0], 'concat': [0.0, 1.0, 0.0], 'max_pooling': [0.0, 0.0, 1.0]},
            'pool': {'global_add_pool': [1.0, 0.0, 0.0], 'global_mean_pool': [0.0, 1.0, 0.0],
                     'global_max_pool': [0.0, 0.0, 1.0]}
        }
        model_str_list = oper_str.split(';')
        layer_str = {'agg': ['', ''],
                     'combine': ['', ''],
                     'act': ['', ''],
                     'layer_connect': ['', ''],
                     'layer_agg': ['']}
        layer_str['layer_agg'][0] = model_str_list[2].split('layer_agg:')[1]
        for j in range(2):
            layer_str['agg'][j] = model_str_list[j].split('agg:')[1].split(',')[0]
            layer_str['combine'][j] = model_str_list[j].split('combine:')[1].split(',')[0]
            layer_str['act'][j] = model_str_list[j].split('act:')[1].split(',')[0]
            layer_str['layer_connect'][j] = model_str_list[j].split('layer_connect:')[1].split('}')[0]
        model = autogel_getmodel(in_features, out_features,gpu=gpu, seed=10)

        model.searched_arch_op = layer_str
        for key in model.searched_arch_op.keys():
            model.searched_arch_z[key] = [op2z_mapping[key][k] for k in model.searched_arch_op[key]]
        model.searched_arch_z = dict(model.searched_arch_z)
        model.searched_arch_op = dict(model.searched_arch_op)
        return model

    def prompt_ger(self, data_name, stage=0):
        if stage == 0:
            return prompt_autogel(data_name, stage=stage)
        arch_list = []
        acc_list = []
        for key in self.arch_dict.keys():
            arch_list.append(key)
            acc_list.append(self.arch_dict[key]['valid_perf'])
        return prompt_autogel(data_name, arch_list=arch_list,acc_list=acc_list, stage=stage)

class autogel_trainer(BaseTrainer):
    def __init__(self, SeachSpace, LLM, gpu=0,file_address='./history/GNAS_LLM/autogel', dataname='Cora'):
        super().__init__(SeachSpace, LLM, file_address)
        self.dataname=dataname
        self.gpu=gpu
        self.device= torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu')

    def get_arch(self,data):
        in_features = max(data.num_node_features, 1)
        out_features = data.num_classes
        model = self.SeachSpace.get_gnn(self.arch, in_features, out_features)

        parser = argparse.ArgumentParser('Interface for evaluate autoGEL')
        parser.add_argument('--task', type=str, default='node', help='type of task', choices=['node', 'graph'])
        parser.add_argument('--dataset', type=str, default='Cora',
                            help='dataset name')  # choices=['PROTEINS', 'IMDB-BINARY', 'IMDB-MULTI', 'BZR', 'COX2', 'DD', 'ENZYMES', 'NCI1']
        parser.add_argument('--train_ratio', type=float, default=0.6, help='ratio of the train against whole')
        parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio of the val against whole')
        parser.add_argument('--test_ratio', type=float, default=0.2, help='ratio of the test against whole')
        parser.add_argument('--bs', type=int, default=128, help='minibatch size')
        parser.add_argument('--gpu', type=int, default=0, help='gpu id')
        parser.add_argument('--optimizer', type=str, default='adam', help='optimizer to use')
        parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
        parser.add_argument('--l2', type=float, default=1e-4, help='l2 regularization weight')
        parser.add_argument('--metric', type=str, default='acc', help='metric for evaluating performance',
                            choices=['acc', 'auc'])
        parser.add_argument('--retrain_epoch', type=int, default=200, help='number of epochs to retrain')
        parser.add_argument('--clip', type=float, default=0.1, help='gradient clipping')
        args = parser.parse_args()

        train_mask, val_mask, test_mask = reset_mask(data, args=args, stratify=None)
        train_loader, val_loader, test_loader = get_loader(data, args=args)
        model, results = retrain(model, train_loader, val_loader, test_loader, train_mask, val_mask, test_mask, args)
        print(self.arch)
        self.model= model
        return model


    def res_to_str(self,res):
        input_lst = res.split('Model:')
        Arch = []
        for i in range(1, len(input_lst)):
            start_index = input_lst[i].find('{')
            end_index = input_lst[i].rfind('}')
            if start_index != -1 and end_index != -1:
                model_str = input_lst[i][start_index + 1:end_index]
            model_str = model_str.replace(" ", "")
            Arch.append(model_str)
        return Arch

    def test(self,dataloader,datamask,dataname='Cora'):
        data=dataloader.dataset
        in_features = max(data.num_node_features, 1)
        out_features = data.num_classes
        # model=self.get_arch(data)
        model=self.model

        parser = argparse.ArgumentParser('Interface for test autoGEL')
        parser.add_argument('--task', type=str, default='node', help='type of task', choices=['node', 'graph'])
        parser.add_argument('--dataset', type=str, default=dataname,help='dataset name')
        args = parser.parse_args()

        test_loss, test_acc, test_auc = eval_model(model, dataloader, datamask, self.device, args, split='test')
        return test_loss, test_acc, test_auc

    def predict(self,data):
        dataloader = DataLoader(data, batch_size=128, shuffle=True)
        in_features = max(data.num_node_features, 1)
        out_features = data.num_classes
        #model=self.get_arch(data)
        model=self.model

        device=self.device
        model.eval()
        predictions = []
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(device)
                prediction = model(batch)
                predictions.append(prediction)
            predictions = torch.cat(predictions, dim=0)
        return predictions



