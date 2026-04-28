import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from tqdm import tqdm
import random  # 新增：用于随机注入噪声

# 确保文件名匹配（如果是 improve_model.py 则用这个名字）
from improve_model import ImprovedEdgeNet

# ==========================================
# 1. 数据集加载器 (Dataset) - 增加抗噪数据增强
# ==========================================
class EdgeDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_size=512):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])

    def __len__(self):
        return len(self.images)

    def add_random_noise(self, img):
        """核心改进：在线随机注入噪声（给模型打抗噪疫苗）"""
        # 50% 的概率保持干净原图，50% 的概率加噪 (保证模型既能学干净的，也能学带噪的)
        if random.random() < 0.5:
            return img
            
        noise_type = random.choice(['gaussian', 'salt_pepper'])
        
        if noise_type == 'gaussian':
            # 随机生成不同强度的高斯噪声 (std从10到30)
            std = random.uniform(10, 30)
            noise = np.random.normal(0, std, img.shape).astype(np.float32)
            noisy_img = cv2.add(img.astype(np.float32), noise)
            return np.clip(noisy_img, 0, 255).astype(np.uint8)
            
        else: # salt_pepper (椒盐噪声)
            noisy_img = np.copy(img)
            prob = random.uniform(0.01, 0.05) # 1% 到 5% 的椒盐噪声
            rnd = np.random.rand(*noisy_img.shape[:2])
            if len(noisy_img.shape) == 3:
                noisy_img[rnd < prob/2] = [0, 0, 0]
                noisy_img[(rnd >= prob/2) & (rnd < prob)] = [255, 255, 255]
            else:
                noisy_img[rnd < prob/2] = 0
                noisy_img[(rnd >= prob/2) & (rnd < prob)] = 255
            return noisy_img

    def preprocess_image(self, img):
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 高斯滤波去噪
        blurred = cv2.GaussianBlur(img, (0, 0), 3)
        # 高通滤波提取细节 (对应开题报告中的自适应预处理)
        highpass = img.astype(np.float32) - blurred.astype(np.float32)
        highpass /= 128.0
        
        max_val = np.max(np.abs(highpass))
        if max_val > 0:
            highpass /= max_val
        return highpass

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace('.jpg', '.png'))

        # 读取图像
        img = cv2.imread(img_path)
        img = cv2.resize(img, (self.img_size, self.img_size))
        
        # 👉 【核心改动点】：在进行滤波和提取特征前，先随机注入噪声
        img = self.add_random_noise(img)
        
        img = self.preprocess_image(img)
        img = np.expand_dims(img, axis=0)

        # 读取标签
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None: # 防错处理
            mask = np.zeros((self.img_size, self.img_size))
        else:
            mask = cv2.resize(mask, (self.img_size, self.img_size))
            
        mask = mask.astype(np.float32) / 255.0
        mask = (mask > 0.5).astype(np.float32) # 严格二值化
        mask = np.expand_dims(mask, axis=0)

        return torch.from_numpy(img).float(), torch.from_numpy(mask).float()

# ==========================================
# 2. 边缘感知损失函数 (Edge-Aware Loss)
# ==========================================
class EdgeAwareLoss(nn.Module):
    def __init__(self, alpha=0.3): 
        super(EdgeAwareLoss, self).__init__()
        self.alpha = alpha

    def forward(self, pred, target):
        # 避免 BCELoss 的数值溢出
        pred = torch.clamp(pred, 1e-7, 1.0 - 1e-7)
        
        # 1. 动态加权 BCE
        num_pos = torch.sum(target == 1.0).clamp(min=1.0)
        num_neg = torch.sum(target == 0.0).clamp(min=1.0)
        beta = num_neg / (num_pos + num_neg)
        weight_map = torch.where(target == 1.0, beta, 1.0 - beta)
        
        bce_loss = nn.BCELoss(weight=weight_map)(pred, target)
        
        # 2. Dice Loss
        smooth = 1e-5
        intersection = torch.sum(pred * target)
        dice_loss = 1.0 - (2.0 * intersection + smooth) / (torch.sum(pred) + torch.sum(target) + smooth)
        
        return self.alpha * bce_loss + (1.0 - self.alpha) * dice_loss

# ==========================================
# 3. 主训练循环
# ==========================================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 [设备: {device}] 正在启动抗噪优化训练 (在线数据增强)...")

    # 超参数
    batch_size = 4
    num_epochs = 50 
    learning_rate = 1e-3 # 调高的 LR 非常关键

    # 路径配置 (使用您的绝对路径更安全，如果您之前用相对路径没问题，也可保持)
    train_img_dir = "/home/sty/pyfile/sketchKeras_pytorch/data/train/images"
    train_mask_dir = "/home/sty/pyfile/sketchKeras_pytorch/data/train/masks"
    val_img_dir = "/home/sty/pyfile/sketchKeras_pytorch/data/val/images"
    val_mask_dir = "/home/sty/pyfile/sketchKeras_pytorch/data/val/masks"
    os.makedirs("weights", exist_ok=True)

    # 加载器
    train_loader = DataLoader(EdgeDataset(train_img_dir, train_mask_dir), batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(EdgeDataset(val_img_dir, val_mask_dir), batch_size=batch_size, shuffle=False, num_workers=4)

    # 模型实例化
    model = ImprovedEdgeNet(in_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = EdgeAwareLoss(alpha=0.3)
    
    # 学习率策略：前 15 轮猛冲，后面每 15 轮减半
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for imgs, masks in pbar:
            imgs, masks = imgs.to(device), masks.to(device)

            preds = model(imgs)
            
            # 多尺度加权 Loss
            if isinstance(preds, list):
                loss = 0.0
                for i, p in enumerate(preds):
                    # 侧边输出 (Side Outputs) 权重 0.3，融合输出 (Fused) 权重 1.0
                    w = 1.0 if i == len(preds) - 1 else 0.3
                    loss += w * criterion(p, masks)
            else:
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
                preds = model(imgs)
                
                # 验证集也保持一致的 Loss 计算逻辑
                if isinstance(preds, list):
                    loss = sum([ (1.0 if i == len(preds)-1 else 0.3) * criterion(p, masks) for i, p in enumerate(preds)])
                else:
                    loss = criterion(preds, masks)
                val_loss += loss.item()
                
        avg_val = val_loss / len(val_loader)
        scheduler.step()

        print(f"📈 Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {avg_val:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        # 智能保存最佳权重 (改了名字，防止覆盖您之前的模型)
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), "weights/improved_edge_net_robust_best.pth")
            print(f"   🌟 发现更优抗噪模型 (Loss: {best_val_loss:.4f})，已保存。")

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"weights/improved_edge_net_robust_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train()