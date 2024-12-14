# TransReID
We modify the code from [TransReID-SSL](https://github.com/damo-cv/TransReID-SSL) and [TransReID](https://github.com/damo-cv/TransReID). You can refer to them for more details.

## Configs
We add some new config files for training TransReID with our CION pre-trained models at TransReID-ROOT/configs/*.

## Prepare Pre-trained Models 
Please download the CION pre-trained models  and put them into your custom file folder.

## Training

We utilize 1  GPU for training. Please modify the `MODEL.PRETRAIN_PATH` and `DATASET.ROOT_DIR` in the config file. For example,

```bash
python train.py --config_file configs/market/resnet50.yml
```

## Evaluation

After the TransReID fine-tuning, you can run the following command to evaluate the performance.

```bash
python test.py --config_file 'choose which config to test' MODEL.DEVICE_ID "('your device id')" TEST.WEIGHT "('your path of trained checkpoints')"
```

## Citation

If you find this code useful for your research, please cite our paper

```
@inproceedings{
zuo2024crossvideo,
title={Cross-video Identity Correlating for Person Re-identification Pre-training},
author={Jialong Zuo and Ying Nie and Hanyu Zhou and Huaxin Zhang and Haoyu Wang and Tianyu Guo and Nong Sang and Changxin Gao},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024}
}
```
