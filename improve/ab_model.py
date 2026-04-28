import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# --- 注意力模块保持不变 ---
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


# 修改 CBAM 以支持“仅通道”模式
class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7, mode='full'):
        super(CBAM, self).__init__()
        self.mode = mode
        self.ca = ChannelAttention(in_planes, ratio)
        if mode == 'full':
            self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        if self.mode == 'full':
            x = x * self.sa(x)
        return x


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
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


# ==========================================
# 核心修改：支持消融实验的 ImprovedEdgeNet
# ==========================================
class ImprovedEdgeNet(nn.Module):
    def __init__(self, in_channels=1, backbone='mobilenet', attn_type='none'):
        """
        backbone: 'mobilenet' 或 'vgg16'
        attn_type: 'none', 'channel', 'cbam'
        """
        super(ImprovedEdgeNet, self).__init__()
        self.attn_type = attn_type
        self.input_conv = nn.Conv2d(in_channels, 3, kernel_size=3, padding=1)

        if backbone == 'mobilenet':
            mnet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT).features
            self.enc1, self.enc2, self.enc3, self.enc4 = mnet[0:2], mnet[2:4], mnet[4:7], mnet[7:14]
            dims = [16, 24, 32, 96]
        else:  # VGG16 基准
            vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
            self.enc1, self.enc2, self.enc3, self.enc4 = vgg[0:4], vgg[4:9], vgg[9:16], vgg[16:23]
            dims = [64, 128, 256, 512]

        # 注意力层初始化
        self.attns = nn.ModuleList([nn.Identity() for _ in range(3)])
        if attn_type != 'none':
            mode = 'full' if attn_type == 'cbam' else 'channel'
            self.attns = nn.ModuleList([CBAM(d, mode=mode) for d in dims[:3]])

        self.up3 = UpBlock(dims[3] + dims[2], 64)
        self.up2 = UpBlock(64 + dims[1], 32)
        self.up1 = UpBlock(32 + dims[0], 16)

        self.side3 = nn.Conv2d(64, 1, kernel_size=1)
        self.side2 = nn.Conv2d(32, 1, kernel_size=1)
        self.side1 = nn.Conv2d(16, 1, kernel_size=1)
        self.fuse = nn.Conv2d(3, 1, kernel_size=1)

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.input_conv(x)
        e1, e2, e3, e4 = self.enc1(x), self.enc2(self.enc1(x)), self.enc3(self.enc2(self.enc1(x))), self.enc4(
            self.enc3(self.enc2(self.enc1(x))))

        # 应用注意力（如果是 Identity 则无效）
        e1, e2, e3 = self.attns[0](e1), self.attns[1](e2), self.attns[2](e3)

        d3 = self.up3(e4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        s3 = F.interpolate(self.side3(d3), size=(H, W), mode='bilinear', align_corners=True)
        s2 = F.interpolate(self.side2(d2), size=(H, W), mode='bilinear', align_corners=True)
        s1 = F.interpolate(self.side1(d1), size=(H, W), mode='bilinear', align_corners=True)

        fused = self.fuse(torch.cat([s3, s2, s1], dim=1))

        if self.training:
            return [torch.sigmoid(s3), torch.sigmoid(s2), torch.sigmoid(s1), torch.sigmoid(fused)]
        return torch.sigmoid(fused)