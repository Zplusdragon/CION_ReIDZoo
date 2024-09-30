import torch
import torch.utils.data as data
import os
import json
from PIL import Image
from .aug import DataAugmentationDINO
from .sampler_ddp import RandomIdentitySampler_DDP
import torch.distributed as dist

def train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    dino_imgs,pids = zip(*batch)
    dino_imgs = zip(*dino_imgs)
    dino_imgs_batch = [torch.stack(img) for img in dino_imgs]
    pids = torch.tensor(pids, dtype=torch.int64)
    return dino_imgs_batch,pids


class TrainDataset(data.Dataset):
    def __init__(self, image_root, annotation_path, dino_transform=None,):
        with open(annotation_path, 'r', encoding='utf8') as fp:
            self.dataset = json.load(fp)
        self.image_root = image_root
        self.dino_transform = dino_transform
        print("Dataset Samples:{}".format(len(self.dataset)))
        print("Dataset Identities:{}".format(self.dataset[-1]["id"]))

    def load_image(self,file_path):
        image = Image.open(os.path.join(self.image_root, file_path))
        return image

    def __getitem__(self, index):
        data = self.dataset[index]
        image = self.load_image(data["file_path"])
        dino_images = self.dino_transform(image)
        pid = data["id"]
        return dino_images, pid

    def __len__(self):
        return len(self.dataset)


# 获取数据加载器
def get_gradual_train_loader(args, num_instances):
    print("Using DinoTransform!")
    dino_transform = DataAugmentationDINO(args.image_size,args.image_mean,args.image_std,args.crop_size,args.global_crops_scale, args.local_crops_scale, args.local_crops_number)
    train_set = TrainDataset(args.image_root,args.annotation_path,dino_transform)
    mini_batch_size = args.batch_size_per_gpu
    data_sampler = RandomIdentitySampler_DDP(train_set, args.batch_size_per_gpu*dist.get_world_size(), num_instances)
    batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)
    train_loader = torch.utils.data.DataLoader(
            train_set,
            num_workers=args.num_workers,
            batch_sampler=batch_sampler,
            collate_fn=train_collate_fn,
            pin_memory=True,
        )
    return train_loader

