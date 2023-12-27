# GNAS-LLM
The code of paper'[Graph Neural Architecture Search with Large Language Models](https://arxiv.org/abs/2310.01436)'
![image](https://github.com/checkuredu/GNAS-LLM/blob/main/Model.pdf)

## Set Up

- We experiment on CUDA 11.6 and torch 1.13.1.
- Setup up a new conda env and install necessary packages.
    ```bash
    conda create -n gnasllm python=3.8
    pip install -r requirements.txt
    ```

- The directory structure should be:
```
.
|- base
|   |- __init__.py
|   |- base.py
|   |- LLM.py
|
|- contrib
|   |- AutoGEL
|       |- data
|       |- aggregate.py
|       |- anal.py
|       |- AutoGEL_main.py
|       |- log.py
|       |- models.py
|       |- searchspace.py
|       |- train.py
|       |- utils.py
|   |- __init__.py
|   |- GNASLLM_Autogel.py
|   |- GNASLLM_NAS_Bench_Graph.py
|
|- example
|   |- history
|   |- __init__.py
|   |- main_AutoGEL.py
|   |- main_NAS_Bench_Graph.py
|
|- __init__.py
|- README.md
```


## Citation
If you find GNAS-LLM useful in your research or applications, please kindly cite:
```tex
@misc{wang2023graph,
      title={Graph Neural Architecture Search with GPT-4}, 
      author={Haishuai Wang and Yang Gao and Xin Zheng and Peng Zhang and Hongyang Chen and Jiajun Bu},
      year={2023},
      eprint={2310.01436},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
} 
```
