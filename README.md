# 超分辨率 + 图像分类 多任务学习

WoC 寒假课题验收代码仓库。

方向：多任务学习

## 项目简介

将 SRNet（超分辨率）和 ClassifyNet（分类）串联，两个子任务各自预训练好之后，再用联合 Loss 端到端微调。通过多任务学习微调，成功将模糊图分类准确率从 47% 提到了 81%。

## 目录结构

```text
.
├── NetSet.py               # 网络定义 (Encoder, SRNet, ClassifyNet)
├── datasets.py             # 数据集 (DIV2KDataset, CIFAR10Dataset, Fuzz)
├── Test.py                 # PSNR / SSIM 计算
├── task1.ipynb             # Task 1: 超分辨率训练 & 测试
├── task2.ipynb             # Task 2: 分类训练 & 测试
├── multi_learning.ipynb    # 多任务联合微调 & 对比
├── DS/                     # 数据集
│   ├── CIFAR10/            #   CIFAR-10
│   ├── DIV2K/train/        #   DIV2K 训练集
│   └── Set14/              #   Set14 测试集
└── results/                # 输出
    ├── task1/              #   SR 权重 + 对比图
    ├── task2/              #   分类权重 + 可视化
    └── multi/              #   联合微调权重
```

## 环境

- Python >= 3.8
- PyTorch >= 2.0, torchvision >= 0.15
- NumPy, Pillow, Matplotlib, Jupyter

```bash
# conda
conda create -n mtl python=3.10 -y && conda activate mtl
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
pip install numpy pillow matplotlib jupyter

# 或者直接 pip
pip install -r requirements.txt
```

## 网络架构

| 模块 | 结构 | 说明 |
| :--- | :--- | :--- |
| Encoder | Conv2d(3→64) + ReLU | 浅层特征提取，两个子网络各自有一份 |
| SRNet | Encoder → 8×ResBlock(128) → PixelShuffle(2×) | 超分辨率，2 倍放大，L1 Loss |
| ClassifyNet | Encoder → Conv×5 + Pool + Dropout → FC(→10) | CIFAR-10 十分类，CE Loss |
| MultiTaskNet | SRNet → Normalize → ClassifyNet | 串起来，联合微调 |

详细定义见 `NetSet.py`。

## 训练流程

分三步走：**预训练 → 组装 → 联合微调**。

### 1. 独立预训练

|  | Task 1 (超分辨率) | Task 2 (分类) |
| :--- | :--- | :--- |
| Notebook | `task1.ipynb` | `task2.ipynb` |
| 数据集 | DIV2K (RandomCrop 128, Bicubic 2× 降采样) | CIFAR-10 |
| Loss | L1 | CrossEntropy |
| 优化器 | Adam, lr=1e-3 | SGD, lr=1e-3, momentum=0.9 |
| 调度 | StepLR (step=25, γ=0.7) | 无 |
| Epochs | 100 | 100 |

### 2. 级联组装

把两个预训练权重分别加载到 `MultiTaskNet.sr` 和 `MultiTaskNet.classifier`。

### 3. 联合微调

在 CIFAR-10 上进行联合训练，图像退化流程：

```
原图 (32×32) → denormalize → Fuzz(Bicubic 缩到 16×16) → MultiTaskNet
```


联合 Loss：

$$\mathcal{L} = \lambda_{SR} \cdot L1(X_{restored},\ X_{clean}) + \lambda_{CLS} \cdot CE(Y_{pred},\ Y_{true})$$

核心思路：为了降低总Loss，迫使SR模块将图像修复到”足够让分类器看清“的程度，从而把高分类准确率。同时梯度回传到SRNet，使其学会提取对分类最有价值的特征。
## 运行顺序

数据集已下载到DS中， 无需额外下载。按照task1 -> task2 -> multi-learning顺序运行即可。仓库也已提供三个任务的预训练模型（见results），可快速开始。

## 实验结果

### 超分辨率 (Set14)

| 指标 | 值 |
| :--- | :--- |
| PSNR | 28.80 dB |
| SSIM | 0.8691 |

### 分类对比 (CIFAR-10)

| 场景 | 输入 | 模型 | 准确率 |
| :--- | :--- | :--- | :--- |
| 原图直接分类 | 清晰 32×32 | ClassifyNet | 84.80% |
| 模糊图直接分类 | 16→32 Bicubic | ClassifyNet | 47.26% |
| 模糊图先修复再分类 | 16×16 | MultiTaskNet | **80.95%** |

模糊图场景下准确率提升 33.69% ，多任务学习有效提升了模糊图片识别的准确率

## 总结


1. 熟悉了 `Conv2d`、`PixelShuffle`、残差连接等基本模块
2. 理解 `Dataset` / `DataLoader` 的数据管线
3. 完成了 预训练 → 权重迁移 → 联合微调 的完整流程
4. 多次S调整shape Mismatch 的 bug ，深入理解张量的维度变化
