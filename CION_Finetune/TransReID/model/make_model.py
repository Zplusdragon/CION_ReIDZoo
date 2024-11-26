import torch
import torch.nn as nn
from .backbones.ResNet import resnet50,resnet18,resnet101,resnet152
from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss
from .backbones.Vision_Transformer import vit_small,vit_base,vit_tiny
from .backbones.Swin_Transformer import swin_tiny,swin_small,swin_base
from .backbones.RepViT import repvit_m1_0,repvit_m1_5,repvit_m2_3,repvit_m0_9
from .backbones.ResNet_IBN import resnet50_ibn_a,resnet101_ibn_a,resnet152_ibn_a,resnet18_ibn_a,resnet34_ibn_a
from .backbones.GhostNet import ghostnet_0_5,ghostnet_1_0,ghostnet_1_3
from .backbones.ConvNext import convnext_base,convnext_tiny,convnext_small
from .backbones.EdgeNext import edgenext_small,edgenext_base,edgenext_x_small
from .backbones.VOLO import volo_d1,volo_d2,volo_d3
from .backbones.Standard_ViT import standard_vit_base,standard_vit_tiny,standard_vit_small
from .backbones.FastViT import fastvit_s12,fastvit_sa12,fastvit_sa24

def shuffle_unit(features, shift, group, begin=1):

    batchsize = features.size(0)
    dim = features.size(-1)
    # Shift Operation
    feature_random = torch.cat([features[:, begin-1+shift:], features[:, begin:begin-1+shift]], dim=1)
    x = feature_random
    # Patch Shuffle Operation
    try:
        x = x.view(batchsize, group, -1, dim)
    except:
        x = torch.cat([x, x[:, -2:-1, :]], dim=1)
        x = x.view(batchsize, group, -1, dim)

    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, dim)

    return x

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class build_resnet(nn.Module):
    def __init__(self, num_classes, cfg,factory):
        super(build_resnet, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.reduce_feat_dim = cfg.MODEL.REDUCE_FEAT_DIM
        self.feat_dim = cfg.MODEL.FEAT_DIM
        self.dropout_rate = cfg.MODEL.DROPOUT_RATE

        self.in_planes = 2048
        self.base = factory[cfg.MODEL.NAME]()
        print('using {} as a backbone'.format(cfg.MODEL.NAME))

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes
        if self.reduce_feat_dim:
            self.fcneck = nn.Linear(self.in_planes, self.feat_dim, bias=False)
            self.fcneck.apply(weights_init_xavier)
            self.in_planes = cfg.MODEL.FEAT_DIM

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate)

        if pretrain_choice == 'self':
            self.base.load_param(model_path)
            print('Loading pretrained ReIDNets model......from {}'.format(model_path))

        else:
            raise ValueError("pretrain choice error!")

    def forward(self, x, label=None, **kwargs):  # label is unused if self.cos_layer == 'no'
        x = self.base(x)
        global_feat = nn.functional.avg_pool2d(x, x.shape[2:4])
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
        if self.reduce_feat_dim:
            global_feat = self.fcneck(global_feat)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)
        if self.dropout_rate > 0:
            feat = self.dropout(feat)

        if self.training:
            if self.cos_layer:
                cls_score = self.arcface(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cls_score, global_feat
        else:
            if self.neck_feat == 'after':
                return feat
            else:
                return global_feat



class build_model(nn.Module):
    def __init__(self, num_classes,cfg, factory):
        super(build_model, self).__init__()

        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.reduce_feat_dim = cfg.MODEL.REDUCE_FEAT_DIM
        self.feat_dim = cfg.MODEL.FEAT_DIM
        self.dropout_rate = cfg.MODEL.DROPOUT_RATE
        print('using {} as a backbone'.format(cfg.MODEL.NAME))
        self.base = factory[cfg.MODEL.NAME](image_size=cfg.INPUT.SIZE_TRAIN)
        self.in_planes = self.base.embed_dim

        if pretrain_choice == 'self':
            self.base.load_param(model_path)
            print('Loading pretrained ReIDNets model checkpoint from {}'.format(model_path))
        else:
            raise ValueError("Must be self!")

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            if self.reduce_feat_dim:
                self.fcneck = nn.Linear(self.in_planes, self.feat_dim, bias=False)
                self.fcneck.apply(weights_init_xavier)
                self.in_planes = cfg.MODEL.FEAT_DIM
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.dropout = nn.Dropout(self.dropout_rate)


    def forward(self, x, label=None, cam_label= None, view_label=None):
        global_feat = self.base(x)
        if self.reduce_feat_dim:
            global_feat = self.fcneck(global_feat)
        feat = self.bottleneck(global_feat)
        feat_cls = self.dropout(feat)

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat_cls, label)
            else:
                cls_score = self.classifier(feat_cls)

            return cls_score, global_feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return global_feat




__factory_T_type = {
    'vit_tiny':vit_tiny,
    'vit_small':vit_small,
    'vit_base':vit_base,
    'resnet18':resnet18,
    'resnet50':resnet50,
    'resnet101':resnet101,
    'resnet152':resnet152,
    'resnet18_ibn':resnet18_ibn_a,
    'resnet50_ibn':resnet50_ibn_a,
    'resnet101_ibn':resnet101_ibn_a,
    'resnet152_ibn':resnet152_ibn_a,
    'swin_tiny':swin_tiny,
    'swin_small':swin_small,
    'swin_base':swin_base,
    'repvit_m1_0':repvit_m1_0,
    'repvit_m1_5':repvit_m1_5,
    'repvit_m0_9':repvit_m0_9,
    'ghostnet_0_5':ghostnet_0_5,
    'ghostnet_1_0':ghostnet_1_0,
    'ghostnet_1_3':ghostnet_1_3,
    'edgenext_x_small':edgenext_x_small,
    'edgenext_small':edgenext_small,
    'edgenext_base':edgenext_base,
    'convnext_small':convnext_small,
    'convnext_tiny':convnext_tiny,
    'convnext_base':convnext_base,
    'standard_vit_tiny':standard_vit_tiny,
    'standard_vit_small':standard_vit_small,
    'standard_vit_base':standard_vit_base,
    'volo_d1':volo_d1,
    'volo_d2':volo_d2,
    'volo_d3':volo_d3,
    'fastvit_s12':fastvit_s12,
    'fastvit_sa12':fastvit_sa12,
    'fastvit_sa24':fastvit_sa24

}

def make_model(cfg, num_class):
    model = build_model(num_class, cfg, __factory_T_type)
    print('===========building model ===========')
    return model
