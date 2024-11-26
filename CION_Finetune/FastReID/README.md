# FastReID with modification for our MGN finetuning.

We utilize the modified fast-reid code from [LUP](https://github.com/DengpanFu/LUPerson) to implement the MGN fine-tuning task. Please follow [fast-reid](https://github.com/JDAI-CV/fast-reid)'s instruction to install fast-reid.

## Configs

We add some new config files for training MGN with our CION pre-trained models at FastReID-ROOT/configs/CMDM/\*.

## Prepare Pre-trained Models 

Please download the CION pre-trained models and put them into your custom file folder.

## Training

We utilize 4 GPU for training. Please modify the `MODEL.PRETRAIN_PATH` and `DATASETS.ROOT` in the config file. For example,

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train_net.py --num-gpus 4 --config-file configs/CMDM/market1501_mgn_R50_cion.yml 
```

## Evaluation

After the MGN fine-tuning, you can run the following command to evaluate the performance.

```
python tools/train_net.py --eval-only --config-file configs/CMDM/market1501_mgn_R50_cion.yml"
```

## Citation

If you find this code useful for your research, please cite the paper

```
@article{zuo2024cross,
  title={Cross-video Identity Correlating for Person Re-identification Pre-training},
  author={Zuo, Jialong and Nie, Ying and Zhou, Hanyu and Zhang, Huaxin and Wang, Haoyu and Guo, Tianyu and Sang, Nong and Gao, Changxin},
  journal={arXiv preprint arXiv:2409.18569},
  year={2024}
}
```
