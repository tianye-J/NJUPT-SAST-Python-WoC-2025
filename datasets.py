import torch
from torchvision import datasets, transforms
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, dataloader
import os
from PIL import Image


class DIV2KDataset(Dataset):
    def __init__(self, root_dir, crop_size=128, scale_factor=2) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.image_files= [f for f in os.listdir(root_dir)]

        self.crop_size = crop_size          #裁剪为crop_size**2大小的图像
        self.scale_factor = scale_factor    #缩小scale_factor倍

        #处理高清图片
        self.transforms_HR = transforms.Compose([
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
        ])

        #生成低清图片
        self.transforms_LR = transforms.Resize(
            size=(crop_size // scale_factor, crop_size // scale_factor),
            interpolation=transforms.InterpolationMode.BICUBIC  #双三次插值
        )

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):

        #拼出图片路径并打开
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')

        #制作训练样本
        img_HR = self.transforms_HR(image)
        img_LR = self.transforms_LR(img_HR)

        return img_LR, img_HR


class Fuzz:
    def __init__(self, scale_factor=2):
        self.scale_factor = scale_factor

    def __call__(self, img_tensor):
        _, h, w = img_tensor.shape
        small = F.resize(
            img_tensor,
            size=[h // self.scale_factor, w // self.scale_factor],
            interpolation=transforms.InterpolationMode.BICUBIC
        )
        return small


class CIFAR10Dataset(Dataset):
    def __init__(self, root_dir='./DS/CIFAR10', train=True, download=True,
                 mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.train = train
        self.mean = mean
        self.std = std

        #数据预处理
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        #加载CIFAR10数据集
        self.dataset = datasets.CIFAR10(
            root=root_dir,
            train=train,
            download=download,
            transform=self.transform
        )

        #类别名称
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer',
                        'dog', 'frog', 'horse', 'ship', 'truck')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
