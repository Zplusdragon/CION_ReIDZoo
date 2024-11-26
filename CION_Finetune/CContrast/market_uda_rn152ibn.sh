

# ResNet50
CUDA_VISIBLE_DEVICES=0,1,2,3 python cluster_contrast_train_usl.py -b 256 -a resnet_ibn152a -d market1501 --iters 200 --eps 0.6 --self-norm --use-hard --hw-ratio 2 --num-instances 8 -pp /home/ma-user/work/Projects/ReIDNet_Finetune/TransReID/log/bs_exp/ResNet152_IBN_MSMT17/64bs_lr0.0004_ep120_warm20_seed0/resnet152_ibn_120.pth --logs-dir log/msmt2market/resnet152_ibn_cion
