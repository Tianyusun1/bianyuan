import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm

def auto_canny(image, sigma=0.33):
    """
    自适应 Canny 边缘检测：根据图像的中值亮度自动计算高低阈值
    """
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged

def apply_sobel(img):
    """
    Sobel 算子：计算 X 翻向和 Y 方向的梯度幅值
    """
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sobel_x, sobel_y)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return magnitude.astype(np.uint8)

def apply_laplacian(img):
    """
    Laplacian (拉普拉斯) 算子：二阶微分算子
    """
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    abs_laplacian = np.absolute(laplacian)
    normalized = cv2.normalize(abs_laplacian, None, 0, 255, cv2.NORM_MINMAX)
    return normalized.astype(np.uint8)

def run_all_baselines(args):
    print("⚙️  正在启动传统算法 (Canny, Sobel, Laplacian) 一键推理...")
    
    # 定义三个子输出文件夹
    out_dirs = {
        "canny": os.path.join(args.base_output_dir, "canny_baseline"),
        "sobel": os.path.join(args.base_output_dir, "sobel_baseline"),
        "laplacian": os.path.join(args.base_output_dir, "laplacian_baseline")
    }
    
    # 自动创建文件夹
    for d in out_dirs.values():
        os.makedirs(d, exist_ok=True)
    
    valid_exts = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = sorted([f for f in os.listdir(args.input_dir) if os.path.splitext(f)[1].lower() in valid_exts])
    
    if not image_files:
        print(f"⚠️ 未在 {args.input_dir} 找到测试图片！")
        return

    print(f"🔍 开始处理 {len(image_files)} 张测试图片，将同步生成三种基线结果...")

    for img_name in tqdm(image_files, desc="传统算子并行处理中"):
        input_path = os.path.join(args.input_dir, img_name)
        base_img_name = os.path.splitext(img_name)[0] + '.png'
        
        # 1. 读取为灰度图
        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if img is None: 
            continue
            
        # 2. 传统高斯模糊去噪 (保证三个算子基于同样的条件对比)
        blurred = cv2.GaussianBlur(img, (3, 3), 0)
        
        # 3. 同步运行三种传统算子
        edges_canny = auto_canny(blurred)
        edges_sobel = apply_sobel(blurred)
        edges_laplacian = apply_laplacian(blurred)
        
        # 4. 保存结果到对应的文件夹
        cv2.imwrite(os.path.join(out_dirs["canny"], base_img_name), edges_canny)
        cv2.imwrite(os.path.join(out_dirs["sobel"], base_img_name), edges_sobel)
        cv2.imwrite(os.path.join(out_dirs["laplacian"], base_img_name), edges_laplacian)

    print(f"\n✅ 任务完成！基线结果已全部保存至: {args.base_output_dir} 目录下的三个子文件夹中。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="传统边缘检测基线测试 (Canny, Sobel, Laplacian)")
    
    # 您的输入目录
    parser.add_argument("--input_dir", "-i", type=str, 
                        default="/home/sty/pyfile/sketchKeras_pytorch/data/test_noisy/images_salt_pepper", 
                        help="测试集原图目录")
    
    # 您的基础输出目录 (脚本会在这里面自动建3个文件夹)
    parser.add_argument("--base_output_dir", "-o", type=str, 
                        default="/home/sty/pyfile/sketchKeras_pytorch/test_results", 
                        help="基础输出目录")
                        
    args = parser.parse_args()
    run_all_baselines(args)