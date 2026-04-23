import torch
import torch.nn as nn
import torchvision.models as models

class HEDNet(nn.Module):
    """
    经典 HED 模型架构复现 (用于对比实验)
    基于 VGG16，不带注意力机制，保留多个侧边输出
    """
    def __init__(self, in_channels=1):
        super(HEDNet, self).__init__()
        
        self.input_conv = nn.Conv2d(in_channels, 3, kernel_size=3, padding=1)
        
        # 借用 VGG16 的特征提取层
        vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        
        self.conv1 = vgg16[0:4]   # 第1阶段 (返回 64 通道)
        self.conv2 = vgg16[4:9]   # 第2阶段 (返回 128 通道)
        self.conv3 = vgg16[9:16]  # 第3阶段 (返回 256 通道)
        self.conv4 = vgg16[16:23] # 第4阶段 (返回 512 通道)
        
        # HED 的经典设计：侧边输出层 (Side Outputs)
        self.score_dsn1 = nn.Conv2d(64, 1, 1)
        self.score_dsn2 = nn.Conv2d(128, 1, 1)
        self.score_dsn3 = nn.Conv2d(256, 1, 1)
        self.score_dsn4 = nn.Conv2d(512, 1, 1)
        
        # 最终融合权重层
        self.fuse = nn.Conv2d(4, 1, 1)

    def forward(self, x):
        h, w = x.shape[2:]
        x = self.input_conv(x)
        
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        
        # 将各阶段特征图转为单通道边缘图
        s1 = self.score_dsn1(c1)
        s2 = self.score_dsn2(c2)
        s3 = self.score_dsn3(c3)
        s4 = self.score_dsn4(c4)
        
        # 全部上采样到原图尺寸
        s2 = nn.functional.interpolate(s2, size=(h, w), mode='bilinear', align_corners=True)
        s3 = nn.functional.interpolate(s3, size=(h, w), mode='bilinear', align_corners=True)
        s4 = nn.functional.interpolate(s4, size=(h, w), mode='bilinear', align_corners=True)
        
        # 融合输出
        fused = self.fuse(torch.cat([s1, s2, s3, s4], dim=1))
        
        if self.training:
            return [torch.sigmoid(s1), torch.sigmoid(s2), torch.sigmoid(s3), torch.sigmoid(s4), torch.sigmoid(fused)]
        else:
            return torch.sigmoid(fused)

if __name__ == "__main__":
    model = HEDNet()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"HED (VGG-16 Baseline) 参数量: {total_params / 1e6:.2f} M")