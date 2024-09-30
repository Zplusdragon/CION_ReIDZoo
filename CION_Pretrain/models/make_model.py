from models.Vision_Transformer import vit_tiny_p16,vit_small_p16,vit_base_p16
from models.Swin_Transformer import swin_tiny,swin_small,swin_base
from models.ResNet import resnet18,resnet50,resnet101,resnet152
from models.ResNet_IBN import resnet18_ibn_a,resnet50_ibn_a,resnet101_ibn_a,resnet152_ibn_a
from models.ConvNext import convnext_tiny,convnext_small,convnext_base
from models.EdgeNext import edgenext_x_small,edgenext_small,edgenext_base
from models.GhostNet import ghostnet_0_5,ghostnet_1_0,ghostnet_1_3
from models.RepViT import repvit_m0_9,repvit_m1_0,repvit_m1_5
from models.FastViT import fastvit_s12,fastvit_sa12,fastvit_sa24
from models.VOLO import volo_d1,volo_d2,volo_d3

model_zoo = {
    'vit_tiny': vit_tiny_p16,
    'vit_small': vit_small_p16,
    'vit_base': vit_base_p16,
    'swin_tiny': swin_tiny,
    'swin_small': swin_small,
    'swin_base': swin_base,
    'resnet18': resnet18,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'resnet18_ibn': resnet18_ibn_a,
    'resnet50_ibn': resnet50_ibn_a,
    'resnet101_ibn': resnet101_ibn_a,
    'resnet152_ibn': resnet152_ibn_a,
    'convnext_tiny': convnext_tiny,
    'convnext_small': convnext_small,
    'convnext_base': convnext_base,
    'edgenext_x_small': edgenext_x_small,
    'edgenext_small': edgenext_small,
    'edgenext_base': edgenext_base,
    "ghostnet_0_5": ghostnet_0_5,
    "ghostnet_1_0": ghostnet_1_0,
    "ghostnet_1_3": ghostnet_1_3,
    'repvit_m0_9': repvit_m0_9,
    'repvit_m1_0': repvit_m1_0,
    'repvit_m1_5': repvit_m1_5,
    'fastvit_s12': fastvit_s12,
    'fastvit_sa12': fastvit_sa12,
    'fastvit_sa24': fastvit_sa24,
    'volo_d1': volo_d1,
    'volo_d2': volo_d2,
    'volo_d3': volo_d3,
}