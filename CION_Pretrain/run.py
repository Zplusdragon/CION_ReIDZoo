from data.dataloader import get_gradual_train_loader
import argparse
import warnings
warnings.filterwarnings("ignore")
import os
import sys
import torch
from torch import nn
import utils
from models.make_model import model_zoo
from models.Head import DINOHead
from loss import GradualIdentityDINOLoss
import time
import datetime
import json
from pathlib import Path
import math


def main_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_root', type=str, default=dataset/CION_AL')
    parser.add_argument('--annotation_path', type=str,
                        default='dataset/CION_AL/cinoal_annos.json',
                        help='path for test annotation json file')

# ***********************************************************************************************************************
# 设置模型backbone的类型和参数
    parser.add_argument('--image_size', type=list, default=[256, 128])
    parser.add_argument('--image_mean', type=int, default=(0.485, 0.456, 0.406))
    parser.add_argument('--image_std', type=int, default=(0.229, 0.224, 0.225))

    # Model parameters
    parser.add_argument('--arch', default='resnet50_ibn', type=str,
                        help="""Name of architecture to train. For quick experiments with ViTs,
            we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--output_dim', default=65336, type=int, help="""Dimensionality of
            the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=bool,
                        help="""Whether or not to weight normalize the last layer of the DINO head.
            Not normalizing leads to better performance but can make the training unstable.
            In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
            parameter for teacher update. The value is increased to 1 during training with cosine schedule.
            We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=bool,
                        help="Whether to use batch normalizations in projection head (Default: False)")

    # Gradual Strategy Settings
    parser.add_argument('--gradual_epochs', default=[40, 60, 80, 100], type=list)
    parser.add_argument('--gradual_instances', default=[2, 4, 6, 8], type=list)

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
                        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
            Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
            of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
            starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
                        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--optimizer', default='adamw', type=str,
                        choices=['adamw', 'sgd', 'lars'],
                        help="""Type of optimizer. We recommend using adamw with ViTs.""")

    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
            weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
            weight decay. We use a cosine schedule for WD and using a larger decay by
            the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
            gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
            help optimization for larger ViT architectures. 0 for disabling.""")


    parser.add_argument('--use_fp16', type=bool, default=True, help="""Whether or not
            to use half precision for training. Improves training time and memory requirements,
            but can provoke instability and slight decay of performance. We recommend disabling
            mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")

    parser.add_argument('--batch_size_per_gpu', default=120, type=int,
                        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
            during which we keep the output layer fixed. Typically doing so during
            the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
            linear warmup (highest LR used during training). The learning rate is linearly scaled
            with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
                        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
            end of optimization. We use a cosine LR schedule with linear warmup.""")

    parser.add_argument('--crop_size', type=list, default=[128, 64])
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.8, 1.),
                        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.8),
                        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
            distributed training; see https://pytorch.org/docs/stable/distributed.html""")

    parser.add_argument('--output_dir', default="logs/resnet50_ibn", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=5, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument("--num_workers", default=4, type=int)

    args = parser.parse_args()
    return args


# ***********************************************************************************************************************
def train_one_epoch(student, teacher, teacher_without_ddp, Loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
                    fp16_scaler, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for it, (dino_images,pids) in enumerate(metric_logger.log_every(data_loader, 50, header)):

        # update weight decay and learning rate according to their schedule
        #print("rank{}:{}".format(dist.get_rank(),pids))
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            if it >= len(lr_schedule):
                param_group['lr'] = lr_schedule[-1]
            else:
                param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                if it >= len(wd_schedule):
                    param_group['weight_decay'] = wd_schedule[-1]
                else:
                    param_group["weight_decay"] = wd_schedule[it]

        # move to gpu
        dino_images = [im.cuda(non_blocking=True) for im in dino_images]
        pids = pids.cuda()
        # teacher and student forward passes + compute loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            teacher_output = (teacher(dino_images[:2]))
            student_output = student(dino_images)
            dino_ouputs = [student_output,teacher_output]
            dino_loss = Loss(dino_ouputs, epoch)
            loss = dino_loss

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            if it>=len(momentum_schedule):
                m = momentum_schedule[-1]
            else:
                m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(dino_loss=dino_loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
    # ============ 分布式训练设置 ... ============
    args = main_parse_args()
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)

    if not os.path.exists(args.output_dir):
        if torch.distributed.get_rank() == 0:
            os.makedirs(args.output_dir)

    # ============ 创建训练数据加载器 ... ============
    data_loader = get_gradual_train_loader(args,args.gradual_instances[0])
    print("DataLoader with {} instances created!".format(args.gradual_instances[0]))

    # ============ 创建学生/教师模型 ... ============
    if args.arch in model_zoo.keys():
        student = model_zoo[args.arch]()
        teacher = model_zoo[args.arch]()
        embed_dim = student.embed_dim
    else:
        raise f"Unknow architecture: {args.arch}"

    # Wrapper is consisted of a backbone and a head.
    student_head = DINOHead(embed_dim,args.output_dim,use_bn=args.use_bn_in_head)
    teacher_head = DINOHead(embed_dim,args.output_dim,use_bn=args.use_bn_in_head)
    student = utils.MultiCropWrapper(student, student_head)
    teacher = utils.MultiCropWrapper(teacher,teacher_head)
    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ 损失函数 ... ============
    Loss = GradualIdentityDINOLoss(out_dim=args.output_dim,ncrops=args.local_crops_number+2,gepochs=args.gradual_epochs,ginstances=args.gradual_instances).cuda()

    # ============ 优化器 ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        print("Using adamw optimizer!")
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        print("Using sgd optimizer!")
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        print("Using lars optimizer!")
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches

    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()
        print("Using fp16 precision.")

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        Loss=Loss,
    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting DINO training from epoch {}!".format(start_epoch))
    for epoch in range(start_epoch, args.epochs):
        # ============ training one epoch of DINO ... ============
        if epoch in args.gradual_epochs:
            idx = args.gradual_epochs.index(epoch)
            num_instances = args.gradual_instances[idx+1]
            data_loader = get_gradual_train_loader(args,num_instances)
            print("DataLoader with {} instances created!".format(num_instances))


        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, Loss,
                                                           data_loader, optimizer, lr_schedule, wd_schedule,
                                                           momentum_schedule,
                                                           epoch, fp16_scaler, args)

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'Loss': Loss.state_dict()
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()


        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))