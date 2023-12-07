import torch
import json
from datetime import datetime
class BaseSeachSpace:
    def __init__(self, candidate):

        self.arch_dict={}
        self.candidate=candidate

    def get_seachspace(self):

        return self.candidate

    def prompt_ger(self, stage=0):

        raise NotImplementedError()

    def eval(self):

        raise NotImplementedError()

    def get_gnn(self):

        raise NotImplementedError()

    def evaluate(self, arch, data):
        perf = self.eval(arch, data)
        self.arch_dict[arch] = perf
        return perf['valid_perf']

    def find_best(self):
        best_valid = max(self.arch_dict.values(), key=lambda x: x['valid_perf'])
        print(f"Arch: {best_valid.key()}, valid_acc:{best_valid['valid_perf']}, test_acc:{best_valid['test_perf']}")
        return best_valid

class BaseTrainer:
    def __init__(self, SeachSpace, LLM, file_address='./history/GNAS_LLM/'):
        self.SeachSpace = SeachSpace # SS must be a BaseSeachSpace class
        self.LLM = LLM
        self.file_address = file_address

    def fit_benchmark(self, data_name='cora', iterations=10):
        '''_____________________________________________________'''
        '''This paragraph specifies the file storage address:'''
        performance_history = []
        messages_history = []
        now = datetime.now()
        date_string = str(now.strftime("%Y-%m-%d-%H-%M-%S"))
        filename_message = self.file_address+ date_string+data_name + 'messages.json'
        filename_performance = self.file_address+ date_string+data_name + 'performance.json'
        '''___________________________________________________________________________'''

        for iteration in range(iterations):
            prompt = self.SeachSpace.prompt_ger(data_name=data_name,stage=iteration)
            res = self.LLM(prompt)
            # print(res)

            # messages=[prompt,res]
            # messages_history.append(messages)
            # with open(filename_message, 'w') as f:
            #     json.dump(messages_history, f)


            Arch = self.res_to_str(res)

            for ar in Arch:
                perf = self.SeachSpace.evaluate(ar,data_name)

                performance = {
                    'arch': ar,
                    'result': perf
                }
                performance_history.append(performance)
                # with open(filename_performance, 'w') as f:
                #     json.dump(performance_history, f)

        best_arch=max(performance_history, key=lambda x: x['result'])
        self.arch = best_arch['arch']
        return self.arch

    def fit(self, data, data_name='Cora', iterations=10):
        '''_____________________________________________________'''
        '''This paragraph specifies the file storage address:'''
        performance_history = []
        messages_history = []
        now = datetime.now()
        date_string = str(now.strftime("%Y-%m-%d-%H-%M-%S"))
        filename_message = self.file_address+ date_string+data_name + 'messages.json'
        filename_performance = self.file_address+ date_string+data_name + 'performance.json'
        '''___________________________________________________________________________'''

        for iteration in range(iterations):
            prompt = self.SeachSpace.prompt_ger(data_name=data_name,stage=iteration)
            res = self.LLM(prompt)
            # print(res)

            # messages=[prompt,res]
            # messages_history.append(messages)
            # with open(filename_message, 'w') as f:
            #     json.dump(messages_history, f)


            Arch = self.res_to_str(res)

            for ar in Arch:
                perf = self.SeachSpace.evaluate(ar,data)

                performance = {
                    'arch': ar,
                    'result': perf
                }
                performance_history.append(performance)
                # with open(filename_performance, 'w') as f:
                #     json.dump(performance_history, f)

        best_arch=max(performance_history, key=lambda x: x['result'])
        self.arch = best_arch['arch']
        self.model = self.get_arch(data)
        return self.arch

    def res_to_str(self):
        raise NotImplementedError()

    def test(self,data):

        return

    def get_arch(self):

        return

    def predict(self,data):

        return


