import torch
import math
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

class evaluate:
    def __init__(self, img1, img2) -> None:
        # 对齐两个张量的尺寸
        self.img1, self.img2 = self._align_tensors(img1, img2)
    
    def _align_tensors(self, img1, img2):
        """将两个张量裁剪到相同的尺寸"""
        # 获取最小的高度和宽度
        min_h = min(img1.size(2), img2.size(2))
        min_w = min(img1.size(3), img2.size(3))
        
        # 裁剪到相同尺寸
        img1 = img1[:, :, :min_h, :min_w]
        img2 = img2[:, :, :min_h, :min_w]
        
        return img1, img2

    def psnr(self):
        mse = torch.mean((self.img1 - self.img2) ** 2)
    
        if mse == 0:  # 如果完全相同，返回最大值
            return torch.tensor(100.0)
        return 10 * torch.log10(1.0 / mse)

    def ssim(self, window_size=11, size_average=True):
        def gaussian(window_size, sigma):
            gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
            return gauss/gauss.sum()  # 归一化，使权重和为1

        def create_window(window_size, channel):
            _1D_window = gaussian(window_size, 1.5).unsqueeze(1)  # unsqueeze: 增加维度 [11] → [11, 1]
            _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)  # [11,11] → [1,1,11,11]
            window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
            return window

        def _ssim(img1, img2, window, window_size, channel, size_average=True):
            mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
            mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

            # .pow(2): 平方操作，等价于 **2
            mu1_sq = mu1.pow(2)
            mu2_sq = mu2.pow(2)
            mu1_mu2 = mu1*mu2

            # 计算方差和协方差
            sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
            sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
            sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

            # SSIM公式中的常数，防止除零
            C1 = 0.01**2
            C2 = 0.03**2

            # SSIM公式：结合亮度、对比度和结构信息
            ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

            if size_average:
                return ssim_map.mean()  # 返回平均值
            else:
                return ssim_map.mean(1).mean(1).mean(1)  # 按维度平均

        (_, channel, _, _) = self.img1.size()  # 获取通道数
        window = create_window(window_size, channel)
        if self.img1.is_cuda:
            window = window.cuda(self.img1.get_device())
        window = window.type_as(self.img1)
        
        return _ssim(self.img1, self.img2, window, window_size, channel, size_average)