# Learning Graph-Enhanced Commander-Executor for Multi-Agent Navigation

This is a PyTorch implementation of the paper: [Learning Graph-Enhanced Commander-Executor for Multi-Agent Navigation](https://arxiv.org/abs/2302.04094)

Project Website: https://sites.google.com/view/mage-x23

## Training

You could start training with by running `sh train_gridworld.sh` in directory [onpolicy/scripts](onpolicy/scripts). 

## Evaluation

Similar to training, you could run `sh render_mpe.sh` in directory [onpolicy/scripts](onpolicy/scripts) to start evaluation. Remember to set up your path to the cooresponding model, correct hyperparameters and related evaluation parameters. 

## Citation
If you find this repository useful, please cite our [paper](https://arxiv.org/abs/2301.03398):
```
@misc{yang2023learning,
      title={Learning Graph-Enhanced Commander-Executor for Multi-Agent Navigation}, 
      author={Xinyi Yang and Shiyu Huang and Yiwen Sun and Yuxiang Yang and Chao Yu and Wei-Wei Tu and Huazhong Yang and Yu Wang},
      year={2023},
      eprint={2302.04094},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```