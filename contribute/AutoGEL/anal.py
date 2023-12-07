from nas_bench_graph import light_read as lightread
from nas_bench_graph import Arch
import json
import heapq
import numpy as np
dataname = 'pubmed'
def find_top_arch(history,top=1):
    for i in range(len(history)):
        history[i]['p']=(history[i]['result']['test_acc'][0],history[i]['result']['valid_acc'][0])
    perf_list = []
    perf_set = set()
    for d in history:
        p = d['p']
        if p not in perf_set:
            perf_set.add(p)
            perf_list.append(d)
    b = heapq.nlargest(top, perf_list,key=lambda x: x['result']['valid_acc'][0])
    return(b)

def print_top_ave(history,top=1):
    top_n = find_top_arch(history,top)
    top_valid = sum([t['result']['valid_acc'][0] for t in top_n]) / top
    top_test = sum([t['result']['test_acc'][0] for t in top_n]) / top
    print(f'top_{top} valid: {top_valid}, top_{top} test: {top_test}')


with open(dataname + 'random_1', 'r') as f:
    history_random=json.load(f)

print(dataname)
history = [history_random[i] for i in [1,2,3,4,6,8,9,10,14,16,19]]
print_top_ave(history,top=1)
print_top_ave(history,top=2)
print_top_ave(history,top=5)
print_top_ave(history,top=10)
