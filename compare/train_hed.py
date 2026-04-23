import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from tqdm import tqdm

# 导入您之前定义的 HED 模型
# 如果文件在不同文件夹，请注意路径引用
from base_hed import HEDNet

# ==========================================
# 1. 数据集加载器 (保持与您的 improve/train.py 一致)
# ==========================================
class EdgeDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_size=512):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])

    def __len__(self):
        return len(self.images)

    def preprocess_image(self, img):
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 高斯滤波去噪
        blurred = cv2.GaussianBlur(img, (0, 0), 3)
        # 高通滤波提取细节
        highpass = img.astype(np.float32) - blurred.astype(np.float32)
        highpass /= 128.0
        
        max_val = np.max(np.abs(highpass))
        if max_val > 0:
            highpass /= max_val
        return highpass

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        # 匹配您的标签命名习惯 (jpg -> png)
        mask_name = os.path.splitext(img_name)[0] + '.png'
        mask_path = os.path.join(self.mask_dir, mask_name)

        # 读取图像
        img = cv2.imread(img_path)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = self.preprocess_image(img)
        img = np.expand_dims(img, axis=0)

        # 读取标签
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros((self.img_size, self.img_size))
        else:
            mask = cv2.resize(mask, (self.img_size, self.img_size))
            
        mask = mask.astype(np.float32) / 255.0
        mask = (mask > 0.5).astype(np.float32) 
        mask = np.expand_dims(mask, axis=0)

        return torch.from_numpy(img).float(), torch.from_numpy(mask).float()

# ==========================================
# 2. HED 多尺度损失函数
# ==========================================
class HEDLoss(nn.Module):
    def __init__(self):
        super(HEDLoss, self).__init__()

    def forward(self, preds, target):
        """
        preds: HED 返回的列表 [s1, s2, s3, s4, fused]
        target: 标签图
        """
        loss = 0.0
        for p in preds:
            # 避免数值溢出
            p = torch.clamp(p, 1e-7, 1.0 - 1e-7)
            # 计算交叉熵
            bce = nn.BCELoss()(p, target)
            loss += bce
        return loss

# ==========================================
# 3. 主训练循环
# ==========================================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 [设备: {device}] 正在启动 HED 基线模型训练...")

    # 超参数 (建议与您的改进模型训练保持一致以公平对比)
    batch_size = 4
    num_epochs = 40 
    learning_rate = 1e-4 # HED 参数量大，建议初始 LR 稍低

    # 路径地址 (根据您的项目结构设定)
    # 训练集: /home/sty/pyfile/sketchKeras_pytorch/data/train/
    train_img_dir = "/home/sty/pyfile/sketchKeras_pytorch/data/train/images"
    train_mask_dir = "/home/sty/pyfile/sketchKeras_pytorch/data/train/masks"
    val_img_dir = "/home/sty/pyfile/sketchKeras_pytorch/data/val/images"
    val_mask_dir = "/home/sty/pyfile/sketchKeras_pytorch/data/val/masks"
    
    os.makedirs("weights", exist_ok=True)

    # 加载器
    train_loader = DataLoader(EdgeDataset(train_img_dir, train_mask_dir), batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(EdgeDataset(val_img_dir, val_mask_dir), batch_size=batch_size, shuffle=False, num_workers=4)

    # 模型实例化
    model = HEDNet(in_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = HEDLoss()
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [HED Train]")
        
        for imgs, masks in pbar:
            imgs, masks = imgs.to(device), masks.to(device)

            # HED 在训练模式下返回 [s1, s2, s3, s4, fused]
            preds = model(imgs)
            loss = criterion(preds, masks)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                # 评估模式下通常只返回最终融合结果，我们这里需要包装成列表适配 Loss
                pred_fused = model(imgs)
                loss = nn.BCELoss()(torch.clamp(pred_fused, 1e-7, 1.0 - 1e-7), masks)
                val_loss += loss.item()
                
        avg_val = val_loss / len(val_loader)
        scheduler.step()

        print(f"📈 Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), "weights/hed_baseline_best.pth")
            print(f"   🌟 已保存最佳 HED 权重。")

if __name__ == "__main__":
    train()