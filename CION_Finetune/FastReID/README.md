# FastReID with modification for our MGN finetuning.
We utilize the modified fast-reid code from [LUP](https://github.com/DengpanFu/LUPerson) to implement the MGN fine-tuning task. Please follow [fast-reid](https://github.com/JDAI-CV/fast-reid)'s instruction to install fast-reid.

## Configs
We add some new config files for training MGN with our CION pre-trained models at FastReID-ROOT/configs/CMDM/\*.

## Train
For example,
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train_net.py --num-gpus 4 --config-file configs/CMDM/market1501_mgn_R50_cion.yml 
```

## Test
After the MGN fine-tuning, you can run the following command to test the performance.
```
python tools/train_net.py --eval-only --config-file configs/CMDM/market1501_mgn_R50_cion.yml"
```
