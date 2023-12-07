from torch_geometric.datasets import Planetoid
from contribute.GNASLLM_Autogel import *

key = '__openai key__'

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
    # print(res_temp)
    return res_temp


def GNASLLM_autogel(dataname='Cora',iterations=10):
    data = Planetoid(root='/data4/zhengxin/data/Planetoid/',name=dataname)

    GNASLLMSS = autogel_SeachSpace()
    GNASLLMtrainer = autogel_trainer(GNASLLMSS, LLM_GPT)
    best_arch = GNASLLMtrainer.fit(data,iterations=iterations)
    print(best_arch)

    parser = argparse.ArgumentParser('Interface for test autoGEL')
    parser.add_argument('--task', type=str, default='node', help='type of task', choices=['node', 'graph'])
    parser.add_argument('--dataset', type=str, default='Cora',help='dataset name')
    parser.add_argument('--train_ratio', type=float, default=0.6, help='ratio of the train against whole')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio of the val against whole')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='ratio of the test against whole')
    parser.add_argument('--bs', type=int, default=128, help='minibatch size')
    args = parser.parse_args()

    train_mask, val_mask, test_mask = reset_mask(data, args=args, stratify=None)
    train_loader, val_loader, test_loader = get_loader(data, args=args)
    test_loss, test_acc, test_auc=GNASLLMtrainer.test(test_loader,test_mask,dataname='Cora')
    print(test_loss, test_acc, test_auc)

    prediction=GNASLLMtrainer.predict(data)
    print(prediction)
    return prediction

if __name__ == '__main__':
    GNASLLM_autogel(dataname='Cora', iterations=10)
