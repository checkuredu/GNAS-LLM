import torch
def experiments_prompt(arch_list):
    prompt1 = '''\nHere are some experimental results that you can use as a reference:\n'''  # 将 arch_list 和 acc_list 按照 acc_list 的元素大小进行排序
    prompt2 = '''\nThe model you propose should be strictly #different# from the structure of the existing experimental results.#You should not raise the models that are already present in the above experimental results again.#\n'''
    arch_l = list(arch_list.keys())
    acc_l = [arch_list[key] for key in arch_l]

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

class BaseSeachSpace:
    def __init__(self, suffix, evaluate, if_oper=True, if_link=True, if_strategy=True, operation_dict=None,link_prompt=None,strategy=None):

        self.arch_dict={}
        self.if_oper = if_oper
        self.if_link = if_link
        self.if_strategy = if_strategy
        if if_oper:
            self.operation_dict=operation_dict
        if if_link:
            self.link_prompt=link_prompt
        if if_strategy:
            self.strategy=strategy

        self.suffix = suffix
        self.eva_fun=evaluate

    def evaluate(self,arch,data):
        acc = self.eva_fun(arch,data)
        self.arch_dict[tuple(arch)]=acc
        return acc

    def prompt_ger(self, dataname, stage=0):
        prompt1 = '''The task is to choose the best GNN  architecture on a given dataset. The architecture will be trained and tested on {}, and the objective is to maximize model accuracy.\n'''.format(dataname)
        prompt_link=''
        prompt_oper2=''
        prompt_strategy=''
        if self.if_link:
            prompt_link = self.link_prompt
        prompt_oper1 = '''There are {} operations that can be selected, including {}.\n'''.format(len(self.operation_dict),', '.join(self.operation_dict.keys()))
        if self.if_oper:
            prompt_oper2 = '''{}\n'''.format('\n'.join(['''The definition of {} is as follows: {{ {} }}'''.format(key,self.operation_dict[key]) for key in self.operation_dict.keys()]))
        prompt2 = '''Once again, your task is to help me find the optimal architecture while specifying the GNN architecture and experimental dataset. The main difficulty of this task is how to reasonably arrange the selection strategy of the architecture, and each selected architecture corresponds to the highest accuracy. We should select a new architecture to query based on the existing operation lists and their corresponding accuracy,in order to iteratively find the architecture.'''
        if self.if_strategy:
            prompt_strategy = self.strategy[stage]
        if len(self.arch_dict)==0:
            return prompt1 + prompt_link + prompt_oper1 + prompt_oper2+ prompt2 + prompt_strategy + self.suffix
        else:
            prompt_experiment = experiments_prompt(self.arch_dict)
            return prompt1 + prompt_link + prompt_oper1 + prompt_oper2+ prompt2 + prompt_experiment + prompt_strategy + self.suffix

    def find_max(self):
        max_value = max(self.arch_dict.values())
        max_arch={}
        for key, value in self.arch_dict.items():
            if value == max_value:
                print(f"Arch: {key}, Acc: {value}")
                max_arch[key]=value
        return max_arch

class Basetrainer:
    def __init__(self, SS, LLM):
        self.SeachSpace = SS # SS must be a BaseSeachSpace class
        self.LLM=LLM

    def fit(self, data, iterations=10):
        for iteration in range(iterations):
            if iteration<4:
                stage=0
            else:
                stage=1
            prompt = self.SeachSpace.prompt_ger(dataname=data, stage=stage)
            Arch = self.LLM(prompt,data)
            for ar in Arch:
                acc = self.SeachSpace.evaluate(ar,data)
        return self.SeachSpace.find_max()

    def test(self,data):

        return

    def predict(self,data):

        return
