import torch
from torchvision import datasets, transforms
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
        
