import torch
import torch.nn as nn
import torch.nn.functional as F

#Encoder
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.features(x)

#task1模型
class SRNet(nn.Module):
    def __init__(self, num_channels=128, num_residual_blocks=8) -> None:
        super().__init__()

        self.encoder = Encoder()

        self.transition = nn.Sequential(
            nn.Conv2d(64, num_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        res_blocks = []
        for _ in range(num_residual_blocks):
            res_blocks.append(nn.Sequential(
                nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
            ))
        self.res_blocks = nn.ModuleList(res_blocks)
        self.res_relu = nn.ReLU(inplace=True)

        #特征压缩
        self.tail = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        #上采样部分
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 64*4, kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.transition(x)
        for block in self.res_blocks:
            x = self.res_relu(x + block(x))
        x = self.tail(x)
        x = self.upsample(x)
        return x
    


# Task2网络
class ClassifyNet(nn.Module):
    def __init__(self):
        super(ClassifyNet, self).__init__()
        self.encoder = Encoder()

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

        self.dropout_conv = nn.Dropout(0.25)
        self.dropout_fc = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.encoder(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout_conv(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = self.dropout_conv(x)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool(x)
        x = self.dropout_conv(x)

        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = F.relu(self.fc2(x))
        x = self.dropout_fc(x)
        x = self.fc3(x)
        return x

