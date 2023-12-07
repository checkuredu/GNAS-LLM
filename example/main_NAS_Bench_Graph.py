import json
import requests
from contribute.GNASLLM_NAS_Bench_Graph import *
key = 'your openai key'
def LLM_GPT(prompt,model="gpt-3.5-turbo"): # "gpt-4"-0314
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + key
    }
    system_content = '''Please pay special attention to my use of special markup symbols in the content below.The special markup symbols is # # ,and the content that needs special attention will be between #.'''
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt},
    ]
    da = {
        "model": model,
        "messages": messages,
        "temperature": 0
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(da))
        response.raise_for_status()  # 检查请求是否成功
        res = response.json()
    except (requests.HTTPError, json.JSONDecodeError) as e:
        print("JSON parsing error:", e)
    except Exception as e:
        print("Other exceptions:", e)

    res_temp = res['choices'][0]['message']['content']
    return res_temp


def GNASLLM_nasbenchgraph(data='cora',iterations=15):

    GNASLLMSeachSpace = nasgraph_SeachSpace(dataname=data)
    GNASLLMtrainer = nasgraph_trainer(GNASLLMSeachSpace, LLM_GPT,dataname=data)

    best_arch = GNASLLMtrainer.fit_benchmark('cora',iterations=iterations)
    print(best_arch)
    return best_arch

if __name__ == '__main__':
    GNASLLM_nasbenchgraph(data='citeseer', iterations=15)