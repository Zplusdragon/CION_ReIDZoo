# ViT-S+ICS
CUDA_VISIBLE_DEVICES=0,1 python cluster_contrast_train_usl.py -b 256 -a resnet_ibn101a -d msmt17 --iters 200 --eps 0.7 --self-norm --use-hard --hw-ratio 2 --num-instances 8 -pp /home/ma-user/work/Projects/ReIDNet_Finetune/TransReID/log/bs_exp/ResNet101_IBN_Market1501/64bs_lr0.0004_ep120_warm20_seed0/resnet101_ibn_120.pth --logs-dir log/market2msmt/resnet101_ibn_cion

# VIT-S
# CUDA_VISIBLE_DEVICES=0,1 python examples/cluster_contrast_train_usl.py -b 256 -a vit_small -d msmt17 --iters 200 --eps 0.7 --self-norm --use-hard --hw-ratio 2 --num-instances 8 -pp ../../log/transreid/market/vit_small_cfs_lup/transformer_120.pth --logs-dir ../../log/cluster_contrast_reid/market2msmt/vit_small_cfs_lup 
