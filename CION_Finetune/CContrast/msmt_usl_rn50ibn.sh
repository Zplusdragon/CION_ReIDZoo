# ViT-S+ICS
CUDA_VISIBLE_DEVICES=0,1 python cluster_contrast_train_usl.py -b 256 -a resnet_ibn50a -d msmt17 --iters 200 --eps 0.7 --self-norm --use-hard --hw-ratio 2 --num-instances 8 -pp /home/ma-user/work/Projects/ReIDZoo_Checkpoints/ResNet50_IBN.pth --logs-dir log/msmt17/resnet50_ibn_cion --eval-step 50

