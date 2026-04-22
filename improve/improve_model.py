import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# ==========================================
# 1. 注意力机制模块 (CBAM) - 保持不变，它是提精的关键
# ==========================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        return x * self.ca(x) * self.sa(x)

# ==========================================
# 2. 精简版上采样模块
# ==========================================
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 尺寸对齐
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return self.conv(torch.cat([x2, x1], dim=1))

# ==========================================
# 3. 瘦身改进版 (精简尺度 + 侧边融合)
# ==========================================
class ImprovedEdgeNet(nn.Module):
    def __init__(self, in_channels=1):
        super(ImprovedEdgeNet, self).__init__()
        
        self.input_conv = nn.Conv2d(in_channels, 3, kernel_size=3, padding=1)
        
        # 骨干网络 MobileNetV2
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        features = mobilenet.features
        
        # 【精简点】只提取到 enc4 (1/16尺度)，放弃极其稀疏的 1/32 尺度
        self.enc1 = features[0:2]   # 16
        self.enc2 = features[2:4]   # 24
        self.enc3 = features[4:7]   # 32
        self.enc4 = features[7:14]  # 96 (作为 Bottleneck)
        
        # 针对前三个尺度的注意力增强
        self.cbam1 = CBAM(16)
        self.cbam2 = CBAM(24)
        self.cbam3 = CBAM(32)
        
        # 解码器：只需要 3 层即可回到原图分辨率
        self.up3 = UpBlock(96 + 32, 64)
        self.up2 = UpBlock(64 + 24, 32)
        self.up1 = UpBlock(32 + 16, 16)
        
        # 🚀 精简侧边输出：只提取 3 个不同细节深度的边缘图
        self.side3 = nn.Conv2d(64, 1, kernel_size=1)
        self.side2 = nn.Conv2d(32, 1, kernel_size=1)
        self.side1 = nn.Conv2d(16, 1, kernel_size=1)
        
        # 最终融合模块 (3 个侧边图融合)
        self.fuse = nn.Conv2d(3, 1, kernel_size=1)

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.input_conv(x)
        
        # 编码阶段
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3) # 这里的 e4 是 1/16 分辨率的高级特征
        
        # 注意力增强
        e1_att = self.cbam1(e1)
        e2_att = self.cbam2(e2)
        e3_att = self.cbam3(e3)
        
        # 解码阶段 (U-Net 结构)
        d3 = self.up3(e4, e3_att)
        d2 = self.up2(d3, e2_att)
        d1 = self.up1(d2, e1_att)
        
        # 🚀 提取侧边结果并放大回原尺寸
        s3 = F.interpolate(self.side3(d3), size=(H, W), mode='bilinear', align_corners=True)
        s2 = F.interpolate(self.side2(d2), size=(H, W), mode='bilinear', align_corners=True)
        s1 = F.interpolate(self.side1(d1), size=(H, W), mode='bilinear', align_corners=True)
        
        # 最终多尺度融合
        fused = self.fuse(torch.cat([s3, s2, s1], dim=1))
        
        if self.training:
            # 训练时返回 [s3, s2, s1, fused] 共 4 张图用于算 Loss
            return [torch.sigmoid(s3), torch.sigmoid(s2), torch.sigmoid(s1), torch.sigmoid(fused)]
        else:
            return torch.sigmoid(fused)

if __name__ == "__main__":
    model = ImprovedEdgeNet(in_channels=1)
    # 参数量对比：之前是 3M+，现在应该在 1M-2M 之间，更轻了
    total_params = sum(p.numel() for p in model.parameters())
    print(f"精简后模型总参数量: {total_params / 1e6:.2f} M")