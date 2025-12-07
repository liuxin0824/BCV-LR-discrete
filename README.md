# Videos are Sample-Efficient Supervisions: Behavior Cloning from Videos via Latent Representations 


This repository provides the discrete implementation of BCV-LR. For continuous control, see [BCV-LR](https://github.com/liuxin0824/BCV-LR/).

## Instruction

Enter the repository and use conda to create a environment.
```
cd CRPTpro

conda env create -f conda_env.yml
```

Use tmux to create a terminal (optional) and then enter the created conda environment:
```
tmux

conda activate CRPTpro
```


Run the experiments. 

```
python train.py
```
The data collection, pre-training, and downstream RL are all included.



## Citation


```
@inproceedings{
liu2025videos,
title={Videos are Sample-Efficient Supervisions: Behavior Cloning from Videos via Latent Representations},
author={Xin Liu and Haoran Li and Dongbin Zhao},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=cx1KfZerNY}
}
```

## Acknowledgement
This implementation is built on [Proto-RL](https://github.com/denisyarats/proto).
