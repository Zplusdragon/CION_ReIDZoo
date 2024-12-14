# Cluster Contrast for Unsupervised Person Re-Identification
We modify the code from [cluster-contrast-reid](https://github.com/alibaba/cluster-contrast-reid). You can refer to the original repo for more details.


## Prepare Pre-trained Models 
Please download the CION pre-trained models and put them into your custom file folder.

## Training

You can use 2 or 4 GPUs for training. For more parameter configuration, please check **`market_usl_xx.sh`**, **`market_uda_xx.sh`**, **`msmt_usl_xx.sh`** and **`msmt_uda_xx.sh`**.

- Please set **`-pp`** as the file path of the pre-trained model. For UDA ReID, the pre-trained model should be fine-tuned on the source dataset at first.

## Citation

If you find this code useful for your research, please cite the paper

```
@inproceedings{
zuo2024crossvideo,
title={Cross-video Identity Correlating for Person Re-identification Pre-training},
author={Jialong Zuo and Ying Nie and Hanyu Zhou and Huaxin Zhang and Haoyu Wang and Tianyu Guo and Nong Sang and Changxin Gao},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024}
}
```
