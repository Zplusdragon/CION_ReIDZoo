# TransReID
We modify the code from [TransReID-SSL](https://github.com/damo-cv/TransReID-SSL) and [TransReID](https://github.com/damo-cv/TransReID). You can refer to them for more details.

## Requirements

### Installation

```bash
(we use /torch 1.8.0 /torchvision=0.9.0 /timm 0.3.4 /cuda 11.1 / A100 for training and evaluation.
Note that we use torch.cuda.amp to accelerate speed of training which requires pytorch >=1.6)
```
### Prepare Pre-trained Models 
Please download the pre-trained models by PASS and put them into your custom file folder.

## Training

We utilize 1  GPU for training. Please modify the `MODEL.PRETRAIN_PATH` and `OUTPUT_DIR` in the config file.

```bash
python train.py --config_file configs/market/vit_small.yml
```
You can also use 'single_head' and 'mean' to obtain almost the same performance while reducing the dimension.

## Evaluation

```bash
python test.py --config_file 'choose which config to test' MODEL.DEVICE_ID "('your device id')" TEST.WEIGHT "('your path of trained checkpoints')"
```

## Citation

If you find this code useful for your research, please cite our paper

```
@InProceedings{He_2021_ICCV,
    author    = {He, Shuting and Luo, Hao and Wang, Pichao and Wang, Fan and Li, Hao and Jiang, Wei},
    title     = {TransReID: Transformer-Based Object Re-Identification},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {15013-15022}
}
```
