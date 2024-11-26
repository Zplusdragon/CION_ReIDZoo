#resnet50
CUDA_VISIBLE_DEVICES=0,1,2,3 python cluster_contrast_train_usl.py -b 256 -a resnet_ibn101a -d market1501 --iters 200 --eps 0.6 --self-norm --use-hard --hw-ratio 2 --num-instances 8 -pp /home/ma-user/work/Projects/ReIDZoo_Checkpoints/ResNet101_IBN.pth --logs-dir log/market/resnet101_ibn_cion
