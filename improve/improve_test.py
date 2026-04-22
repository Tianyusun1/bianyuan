import os
import torch
import cv2
import numpy as np
import argparse
from tqdm import tqdm

# 从你最新的瘦身版模型文件中导入
from improve_model import ImprovedEdgeNet

# ==========================================
# 1. 预处理函数 (必须与 train.py 严格一致)
# ==========================================
def preprocess(img, img_size=512):
    """
    步骤：缩放 -> 灰度化 -> 高斯去噪 -> 高通滤波 -> 归一化 -> 填充
    """
    h, w = img.shape[:2]
    # 等比例缩放
    if w > h:
        new_width, new_height = (img_size, int(img_size / w * h))
    else:
        new_width, new_height = (int(img_size / h * w), img_size)
    resized_img = cv2.resize(img, (new_width, new_height))

    # 转灰度
    if len(resized_img.shape) == 3:
        gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = resized_img

    # 自适应滤波模块 (高斯去噪 + 高通滤波)
    blurred = cv2.GaussianBlur(gray, (0, 0), 3)
    highpass = gray.astype(np.float32) - blurred.astype(np.float32)
    highpass /= 128.0
    
    # 归一化：确保输入分布与训练时一致
    max_val = np.max(np.abs(highpass))
    if max_val > 0:
        highpass /= max_val
        
    # 填充到 512x512
    input_tensor = np.zeros((img_size, img_size), dtype=np.float32)
    input_tensor[:new_height, :new_width] = highpass
    
    return input_tensor, (new_height, new_width), (h, w)

# ==========================================
# 2. 后处理函数
# ==========================================
def postprocess(pred, thresh=0.4, smooth=False):
    """
    二值化并处理为图像格式
    """
    # 核心：根据阈值提取边缘
    pred[pred < thresh] = 0.0
    
    # 这里保持黑底白线 (0为背景，255为边缘)，方便 evaluate.py 计算指标
    # 如果你要看白底黑线（像素描一样），请取消下面这行的注释
    # pred = 1.0 - pred 
    
    pred *= 255.0
    pred = np.clip(pred, 0, 255).astype(np.uint8)
    
    if smooth:
        pred = cv2.medianBlur(pred, 3)
        
    return pred

# ==========================================
# 3. 核心批量推理逻辑
# ==========================================
def batch_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 [设备: {device}] 正在初始化推理引擎...")
    
    # 1. 加载瘦身版模型
    model = ImprovedEdgeNet(in_channels=1).to(device)
    
    if os.path.exists(args.weight):
        # 使用 weights_only=True 提高安全性（PyTorch 新版本建议）
        model.load_state_dict(torch.load(args.weight, map_location=device))
        model.eval()  # 必须开启 eval 模式，此时模型只返回 fused 图
        print(f"✅ 权重加载成功: {args.weight}")
    else:
        print(f"❌ 错误: 找不到权重文件 {args.weight}")
        return

    # 2. 目录准备
    os.makedirs(args.output_dir, exist_ok=True)
    
    valid_exts = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = sorted([f for f in os.listdir(args.input_dir) if os.path.splitext(f)[1].lower() in valid_exts])
    
    if not image_files:
        print(f"⚠️ 在 {args.input_dir} 没找到图！")
        return

    print(f"开始处理 {len(image_files)} 张图片...")

    # 3. 推理循环
    for img_name in tqdm(image_files, desc="推理中"):
        input_path = os.path.join(args.input_dir, img_name)
        img = cv2.imread(input_path)
        if img is None: continue

        # 预处理
        processed_img, new_size, orig_size = preprocess(img)
        
        # 转换为 Tensor [1, 1, 512, 512]
        x = torch.from_numpy(processed_img).unsqueeze(0).unsqueeze(0).to(device)

        # 推理
        with torch.no_grad():
            # 瘦身版模型在 eval() 模式下直接返回融合后的 Tensor
            pred_tensor = model(x)
        
        # 后处理
        pred_np = pred_tensor.squeeze().cpu().numpy()
        # 裁掉填充部分
        pred_np = pred_np[:new_size[0], :new_size[1]]
        
        output = postprocess(pred_np, thresh=args.thresh, smooth=args.smooth)
        
        # 还原尺寸
        output_resized = cv2.resize(output, (orig_size[1], orig_size[0]), interpolation=cv2.INTER_CUBIC)

        # 保存为 PNG
        out_name = os.path.splitext(img_name)[0] + ".png"
        cv2.imwrite(os.path.join(args.output_dir, out_name), output_resized)

    print(f"\n🎉 处理完毕！结果保存在: {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="瘦身版模型测试脚本")
    
    # 使用你提供的绝对路径作为默认值
    parser.add_argument("--input_dir", "-i", type=str, 
                        default="/home/sty/pyfile/sketchKeras_pytorch/data/test/images", 
                        help="测试图片目录")
    parser.add_argument("--output_dir", "-o", type=str, 
                        default="test_results/improve", 
                        help="输出结果目录")
    parser.add_argument("--weight", "-w", type=str, 
                        default="/home/sty/pyfile/sketchKeras_pytorch/weights/improved_edge_net_best.pth", 
                        help="权重路径")
    parser.add_argument("--thresh", "-t", type=float, default=0.4, help="阈值")
    parser.add_argument("--smooth", action="store_true", help="开启中值平滑")
    
    args = parser.parse_args()
    batch_inference(args)