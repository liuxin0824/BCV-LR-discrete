# Videos are Sample-Efficient Supervisions: Behavior Cloning from Videos via Latent Representations 


This repository provides the discrete implementation of BCV-LR. For continuous control, see [BCV-LR](https://github.com/liuxin0824/BCV-LR/).



## Data preparation
Download the procgen expert video data [here](https://drive.google.com/drive/folders/1XjpcfOm0NafPYFPnNtoHfhJ4nHVkQSB1) provided by lapo, unzip it, and place it in the expert_data directory. The expert_data dir should look like this:

```
expert_data
   --- starpilot
      --- train
      --- test
   ...
```

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
The implementation and data are built on [LAPO](https://github.com/schmidtdominik/LAPO).
