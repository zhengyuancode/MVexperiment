import torch
import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, channel, ratio=16):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio),  # 压缩通道
            nn.ReLU(),
            nn.Linear(channel // ratio, channel),  # 恢复通道
            nn.Sigmoid()  # 生成权重
        )

    def forward(self, x):
        b, c, _, _ = x.size()  # 批量大小和通道数
        # 全局池化后展平，通过全连接层生成权重，再重塑为(b,c,1,1)
        weight = self.fc(self.gap(x).view(b, c)).view(b, c, 1, 1)
        return x * weight  # 通道加权


class UNet(nn.Module):
    """基线模型：无注意力模块的基础U-Net"""
    def __init__(self, n_classes=28):  # 默认包含背景类（28类）
        super().__init__()
        # 编码器（下采样）
        self.encoder = nn.Sequential(
            # 第1层：3→64，3x3卷积+ReLU，MaxPool下采样（尺寸/2）
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), 
            
            # 第2层：64→128，3x3卷积+ReLU，MaxPool下采样（尺寸/2）
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2) 
        )
        
        # 解码器（上采样）
        self.decoder = nn.Sequential(
            # 第1层：128→64，转置卷积上采样（尺寸×2）
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2), 
            nn.ReLU(),
            
            # 第2层：64→32，转置卷积上采样（尺寸×2）
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2), 
            nn.ReLU(),
            
            # 输出层：32→n_classes，1x1卷积调整通道数
            nn.Conv2d(32, n_classes, kernel_size=1)  
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x) 
        return x


class UNet_Attention(nn.Module):
    """带注意力的改进模型：在编码器和解码器间插入SE注意力模块"""
    def __init__(self, n_classes=28, attention_type="se"):
        super().__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # 注意力模块（SEBlock）
        if attention_type == "se":
            self.attention = SEBlock(channel=128)  
        else:
            self.attention = None 
        
        # 解码器（
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, n_classes, kernel_size=1)
        )

    def forward(self, x):
        x = self.encoder(x) 
        if self.attention:
            x = self.attention(x) 
        x = self.decoder(x) 
        return x