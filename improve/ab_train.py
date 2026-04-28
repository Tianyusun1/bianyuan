import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from ab_model import ImprovedEdgeNet
from train import EdgeDataset, EdgeAwareLoss  # 复用原有的数据类


# 修改 Loss 以支持消融 (BCE vs BCE+Dice)
class AblationLoss(torch.nn.Module):
    def __init__(self, use_dice=True, alpha=0.3):
        super().__init__()
        self.use_dice = use_dice
        self.alpha = alpha
        self.bce = torch.nn.BCELoss()

    def forward(self, pred, target):
        pred = torch.clamp(pred, 1e-7, 1.0 - 1e-7)
        # 动态加权 BCE 逻辑
        num_pos = torch.sum(target == 1.0).clamp(min=1.0)
        num_neg = torch.sum(target == 0.0).clamp(min=1.0)
        beta = num_neg / (num_pos + num_neg)
        self.bce.weight = torch.where(target == 1.0, beta, 1.0 - beta).to(pred.device)

        loss_val = self.bce(pred, target)
        if self.use_dice:
            smooth = 1e-5
            intersection = torch.sum(pred * target)
            dice = 1.0 - (2.0 * intersection + smooth) / (torch.sum(pred) + torch.sum(target) + smooth)
            loss_val = self.alpha * loss_val + (1.0 - self.alpha) * dice
        return loss_val


def run_experiment(exp_name, config):
    print(f"\n{'=' * 20}\n开始实验: {exp_name}\n配置: {config}\n{'=' * 20}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化
    model = ImprovedEdgeNet(in_channels=1, backbone=config['backbone'], attn_type=config['attn']).to(device)
    criterion = AblationLoss(use_dice=config['use_dice'])
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_loader = DataLoader(EdgeDataset("data/train/images", "data/train/masks"), batch_size=4, shuffle=True)
    val_loader = DataLoader(EdgeDataset("data/val/images", "data/val/masks"), batch_size=4, shuffle=False)

    best_loss = float('inf')
    epochs = 30  # 消融实验通常 30 轮足以看出趋势

    for epoch in range(epochs):
        model.train()
        for imgs, masks in tqdm(train_loader, desc=f"{exp_name} Epoch {epoch + 1}"):
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss = sum([(1.0 if i == 3 else 0.3) * criterion(p, masks) for i, p in enumerate(preds)])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                p = model(imgs)
                val_loss += criterion(p, masks).item()

        avg_val = val_loss / len(val_loader)
        if avg_val < best_loss:
            best_loss = avg_val
            torch.save(model.state_dict(), f"weights/best_{exp_name}.pth")
            print(f"--- {exp_name} 保存最佳模型 (Loss: {avg_val:.4f}) ---")


# ==========================================
# 实验主循环
# ==========================================
if __name__ == "__main__":
    os.makedirs("weights", exist_ok=True)

    experiments = {
        "Exp1_Base_MNet": {"backbone": "mobilenet", "attn": "none", "use_dice": False},
        "Exp2_MNet_Chan": {"backbone": "mobilenet", "attn": "channel", "use_dice": False},
        "Exp3_MNet_CBAM": {"backbone": "mobilenet", "attn": "cbam", "use_dice": False},
        "Exp4_MNet_Full": {"backbone": "mobilenet", "attn": "cbam", "use_dice": True},
        "Exp6_Base_VGG": {"backbone": "vgg16", "attn": "none", "use_dice": False},
    }

    for name, cfg in experiments.items():
        run_experiment(name, cfg)