from base import Basetrainer,BaseSeachSpace
import json
import requests
from AutoGEL.AutoGEL_main import main_autogel
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

link_prompt = '''The definition of search space for models follow the neighborhood aggregation schema, i.e., the Message Passing Neural Network (MPNN), which is formulated as:
    {
        $$\mathbf{m}_{v}^{k+1}=AGG_{k}(\{M_{k}(\mathbf{h}_{v}^{k}\mathbf{h}_{u}^{k},\mathbf{e}_{vu}^{k}):u\in N(v)\})$$
        $$\mathbf{h}_{v}^{k+1}=ACT_{k}(COM_{k}(\{\mathbf{h}_{v}^{k},\mathbf{m}_{v}^{k+1}\}))$$
        $$\hat{\mathbf{y}}=R(\{\mathbf{h}_{v}^{L}|v\in G\})$$
    where $k$ denotes $k$-th layer, $N(v)$ denotes a set of neighboring nodes of $v$, $\mathbf{h}_{v}^{k}$, $\mathbf{h}_{u}^{k}$ denotes hidden embeddings for $v$ and $u$ respectively, $\mathrm{e}_{vu}^{k}$ denotes features for edge e(v, u) (optional), $\mathbf{m}_{v}^{k+1}$denotes the intermediate embeddings gathered from neighborhood $N(v)$, $M_k$ denotes the message function, $AGG_{k}$ denotes the neighborhood aggregation function, $COM_{k}$ denotes the combination function between intermediate embeddings and embeddings of node $v$ itself from the last layer, $ACT_{k}$ denotes activation function. Such message passing phase in repeats for $L$ times (i.e.,$ k\in\{1,\cdots,L\}$). For graph-level tasks, it further follows the readout phase in where information from the entire graph $G$ is aggregated through readout function $R(·)$.
    }'''
strategy = [notice1,notice2]
suffix = '''Please answer me as briefly as possible. You should give 10 different model structures at a time.
For example:
Model1 = {layer1:{ agg:sum, combine:concat, act:prelu, layer_connect:skip_cat}; layer2:{ agg:sum, combine:concat, act:relu,layer_connect:stack}; layer_agg:concat}
Model2 = {layer1:{ agg:mean, combine:sum, act:relu, layer_connect:skip_sum}; layer2:{ agg:max, combine:sum, act:prelu, layer_connect:skip_cat}; layer_agg:max_pooling}
Model ...
...
Model10 = {layer1:{ agg:max, combine:sum, act:relu, layer_connect:stack}; layer2:{agg:mean, combine:sum, act:relu, layer_connect:skip_sum}; layer_agg:concat}
And The response you give must strictly follow the format of this example. '''

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
        start_index = input_lst[i].find('{')
        end_index = input_lst[i].rfind('}')
        if start_index != -1 and end_index != -1:
            model_str = input_lst[i][start_index + 1:end_index]
        ans.append(model_str)
    return ans

def eval(operations,dataname):
    model_str_list = operations.split(';')
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
    results = main_autogel(layer_str, dataname=dataname, seed=10, gpu=2)
    return results

GNASLLMSS = BaseSeachSpace(suffix=suffix,evaluate=eval,if_oper=False,link_prompt=link_prompt,strategy=strategy)
GNASLLMtrainer = Basetrainer(GNASLLMSS, LLM)

best_arch = GNASLLMtrainer.fit('cora')
print(best_arch)