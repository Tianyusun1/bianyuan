import os
import scipy.io as sio
import numpy as np
import cv2
import shutil
from tqdm import tqdm

def process_split(split_name):
    """
    处理特定的数据集分支 (train, val, 或 test)
    """
    # 1. 定义原始 BSDS500 数据集路径 (根据你的截图确定)
    bsds_root = os.path.join("data", "BSR", "BSDS500", "data")
    img_dir = os.path.join(bsds_root, "images", split_name)
    gt_dir = os.path.join(bsds_root, "groundTruth", split_name)
    
    # 2. 定义目标输出路径 (用于后续训练和测试)
    # 结构为: data/train/images, data/val/images, data/test/images 等
    out_img_dir = os.path.join("data", split_name, "images")
    out_mask_dir = os.path.join("data", split_name, "masks")
    
    # 创建目标目录
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)
    
    # 检查源目录是否存在
    if not os.path.exists(img_dir):
        print(f"跳过 {split_name}: 找不到源图片目录 {img_dir}")
        return

    # 获取所有图片名称
    img_names = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
    
    print(f"\n开始处理 BSDS500 [{split_name}] 分支，共发现 {len(img_names)} 张图片...")
    
    for img_name in tqdm(img_names):
        base_name = os.path.splitext(img_name)[0]
        
        src_img_path = os.path.join(img_dir, img_name)
        src_gt_path = os.path.join(gt_dir, base_name + '.mat')
        
        dst_img_path = os.path.join(out_img_dir, img_name)
        dst_mask_path = os.path.join(out_mask_dir, base_name + '.png')
        
        # --- 步骤 A: 复制原图 ---
        shutil.copy(src_img_path, dst_img_path)
        
        # --- 步骤 B: 解析并转换 .mat 标签 ---
        if os.path.exists(src_gt_path):
            try:
                mat = sio.loadmat(src_gt_path)
                ground_truth = mat['groundTruth']
                num_annotators = ground_truth.shape[1]
                
                edge_map = None
                for i in range(num_annotators):
                    # 解析 MATLAB 嵌套结构中的 Boundaries 矩阵
                    boundary = ground_truth[0, i][0][0]['Boundaries']
                    
                    # 鲁棒性处理：确保 boundary 是二维矩阵
                    if isinstance(boundary, np.ndarray) and boundary.shape == (1, 1):
                        boundary = boundary[0, 0]
                        
                    if edge_map is None:
                        edge_map = np.zeros_like(boundary, dtype=np.float32)
                    edge_map += boundary
                    
                # 计算多专家的平均边缘概率
                edge_map = edge_map / num_annotators
                
                # 二值化处理：对应开题报告中“精准提取”的要求
                # 阈值设为 0.3 是通用做法，即 30% 以上的专家认为是边缘则保留
                edge_map[edge_map >= 0.3] = 255.0
                edge_map[edge_map < 0.3] = 0.0
                
                # 保存为单通道 PNG 掩膜
                cv2.imwrite(dst_mask_path, edge_map.astype(np.uint8))
            except Exception as e:
                print(f"\n处理 {base_name}.mat 时发生错误: {e}")
        else:
            print(f"\n警告: 找不到 {base_name}.mat 的标签文件")

if __name__ == "__main__":
    # 分别处理三个数据集分支，确保完全隔离
    for split in ["train", "val", "test"]:
        process_split(split)
        
    print("\n" + "="*50)
    print("数据预处理全部完成！")
    print("你的数据现在存放于根目录下的 data/ 文件夹中:")
    print(" - 训练集: data/train/ (用于模型炼丹)")
    print(" - 验证集: data/val/   (用于训练时监控模型性能)")
    print(" - 测试集: data/test/  (用于最终跑指标和出图)")
    print("="*50)