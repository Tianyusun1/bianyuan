import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm

def add_gaussian_noise(image, mean=0, std=30):
    """
    添加高斯噪声
    std 越大，噪声越强。这里设为 30 属于较强的干扰。
    """
    # 随机生成高斯噪声矩阵
    noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    # 将噪声叠加到原图上
    noisy_img = cv2.add(image.astype(np.float32), noise)
    # 限制像素值在 0-255 之间
    return np.clip(noisy_img, 0, 255).astype(np.uint8)

def add_salt_and_pepper_noise(image, prob=0.05):
    """
    添加椒盐噪声
    prob: 噪声比例，0.05 表示有 5% 的像素会被破坏（变成纯黑或纯白）
    """
    noisy_img = np.copy(image)
    # 随机生成一个与图像同尺寸的矩阵，值在 0-1 之间
    rnd = np.random.rand(*noisy_img.shape[:2])
    
    # 区分单通道(灰度图)和三通道(彩色图)
    if len(noisy_img.shape) == 3:
        noisy_img[rnd < prob/2] = [0, 0, 0]        # 椒噪声 (纯黑)
        noisy_img[(rnd >= prob/2) & (rnd < prob)] = [255, 255, 255] # 盐噪声 (纯白)
    else:
        noisy_img[rnd < prob/2] = 0
        noisy_img[(rnd >= prob/2) & (rnd < prob)] = 255
        
    return noisy_img

def generate_noise_datasets(args):
    print("🌪️  正在生成极限环境下的噪声测试集...")
    
    # 定义输出目录
    dir_gaussian = os.path.join(args.output_base, "images_gaussian")
    dir_sp = os.path.join(args.output_base, "images_salt_pepper")
    os.makedirs(dir_gaussian, exist_ok=True)
    os.makedirs(dir_sp, exist_ok=True)
    
    valid_exts = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = sorted([f for f in os.listdir(args.input_dir) if os.path.splitext(f)[1].lower() in valid_exts])
    
    if not image_files:
        print(f"⚠️ 在 {args.input_dir} 中没找到图片，请检查路径！")
        return

    print(f"🔍 找到 {len(image_files)} 张干净测试图，准备注入噪声...")

    for img_name in tqdm(image_files, desc="注入噪声中"):
        img_path = os.path.join(args.input_dir, img_name)
        img = cv2.imread(img_path)
        if img is None: 
            continue
            
        # 1. 生成并保存高斯噪声图
        img_gaussian = add_gaussian_noise(img, std=30)
        cv2.imwrite(os.path.join(dir_gaussian, img_name), img_gaussian)
        
        # 2. 生成并保存椒盐噪声图
        img_sp = add_salt_and_pepper_noise(img, prob=0.05)
        cv2.imwrite(os.path.join(dir_sp, img_name), img_sp)

    print("\n" + "="*50)
    print("✅ 噪声测试集生成完毕！")
    print(f"👉 高斯噪声测试集保存在: {dir_gaussian}")
    print(f"👉 椒盐噪声测试集保存在: {dir_sp}")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="给测试数据集添加高斯和椒盐噪声")
    
    # 输入：您原始的干净测试集原图的路径
    parser.add_argument("--input_dir", type=str, 
                        default="/home/sty/pyfile/sketchKeras_pytorch/data/test/images",
                        help="原始干净测试集路径")
                        
    # 输出：生成的新噪声测试集的主目录
    parser.add_argument("--output_base", type=str, 
                        default="/home/sty/pyfile/sketchKeras_pytorch/data/test_noisy",
                        help="噪声数据集的保存基准路径")
                        
    args = parser.parse_args()
    
    generate_noise_datasets(args)