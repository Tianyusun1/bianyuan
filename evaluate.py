import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt

def evaluate_dataset(pred_dir, gt_dir, num_thresh=99, tolerance=2):
    pred_files = [f for f in os.listdir(pred_dir) if f.endswith(('.png', '.jpg'))]
    
    if not pred_files:
        print("❌ 未在预测文件夹中找到图片！")
        return

    print(f"🔍 开始权威评估 (ODS/OIS)，共找到 {len(pred_files)} 张预测图片...")
    print(f"⚙️  设置: 阈值数量 = {num_thresh}, 像素容差 = {tolerance}")

    # 生成 0.01 到 0.99 的阈值数组
    thresholds = np.linspace(0.01, 0.99, num_thresh)
    
    # 用于记录整个数据集在不同阈值下的累加值
    dataset_tp_pred = np.zeros(num_thresh)
    dataset_tp_gt = np.zeros(num_thresh)
    dataset_pred_sum = np.zeros(num_thresh)
    dataset_gt_sum = np.zeros(num_thresh)
    
    total_mae = 0.0
    img_best_f1s = []  # 用于计算 OIS (每张图最优 F1 的平均值)
    valid_count = 0

    for pred_file in tqdm(pred_files, desc="计算指标"):
        pred_path = os.path.join(pred_dir, pred_file)
        gt_path = os.path.join(gt_dir, os.path.splitext(pred_file)[0] + '.png')
        
        if not os.path.exists(gt_path):
            continue
            
        # 1. 读取并归一化
        pred_img = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        pred_prob = pred_img.astype(np.float32) / 255.0
        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        
        if pred_img.shape != gt_img.shape:
            pred_prob = cv2.resize(pred_prob, (gt_img.shape[1], gt_img.shape[0]))
            
        # 2. 计算 MAE
        gt_norm = (gt_img > 0).astype(np.float32)
        total_mae += np.mean(np.abs(pred_prob - gt_norm))
        
        # 3. 对 GT 进行一次性膨胀 (节省时间)
        gt_binary = (gt_img > 0).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tolerance*2+1, tolerance*2+1))
        gt_dilated = cv2.dilate(gt_binary, kernel)
        
        img_f1_list = np.zeros(num_thresh)

        # 4. 遍历所有阈值
        for i, t in enumerate(thresholds):
            pred_binary = (pred_prob >= t).astype(np.uint8)
            pred_dilated = cv2.dilate(pred_binary, kernel)
            
            # 计算当前阈值下的 TP 等基础指标
            tp_pred = np.sum(np.logical_and(pred_binary == 1, gt_dilated == 1))
            tp_gt = np.sum(np.logical_and(gt_binary == 1, pred_dilated == 1))
            sum_p = np.sum(pred_binary)
            sum_g = np.sum(gt_binary)
            
            # 累加到全局
            dataset_tp_pred[i] += tp_pred
            dataset_tp_gt[i] += tp_gt
            dataset_pred_sum[i] += sum_p
            dataset_gt_sum[i] += sum_g
            
            # 计算当前图片的 F1 (用于 OIS)
            p = tp_pred / sum_p if sum_p > 0 else 0
            r = tp_gt / sum_g if sum_g > 0 else 0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            img_f1_list[i] = f1
            
        img_best_f1s.append(np.max(img_f1_list))
        valid_count += 1
        
    if valid_count == 0:
        print("❌ 未能匹配到任何真实标签，请检查 --gt_dir 路径！")
        return

    # ==========================================
    # 计算最终评估指标
    # ==========================================
    # 避免除以零
    P = np.divide(dataset_tp_pred, dataset_pred_sum, out=np.zeros_like(dataset_tp_pred), where=dataset_pred_sum!=0)
    R = np.divide(dataset_tp_gt, dataset_gt_sum, out=np.zeros_like(dataset_tp_gt), where=dataset_gt_sum!=0)
    F1 = np.divide(2 * P * R, P + R, out=np.zeros_like(P), where=(P+R)!=0)
    
    # 寻找最佳全局阈值 (ODS)
    ods_index = np.argmax(F1)
    ods_f1 = F1[ods_index]
    ods_p = P[ods_index]
    ods_r = R[ods_index]
    best_thresh = thresholds[ods_index]
    
    # 计算 OIS 和平均 MAE
    ois_f1 = np.mean(img_best_f1s)
    avg_mae = total_mae / valid_count

    print("\n" + "🏆 最终权威评估指标 🏆".center(40, "="))
    print(f"📍 最佳全局阈值 (ODS Thresh): {best_thresh:.2f}")
    print(f"📊 ODS F1-Score (全数据集最优): {ods_f1:.4f}")
    print(f"📊 OIS F1-Score (单图自适应最优): {ois_f1:.4f}")
    print(f"🎯 ODS Precision (精确率):    {ods_p:.4f}")
    print(f"🧲 ODS Recall    (召回率):    {ods_r:.4f}")
    print(f"📉 MAE           (平均误差):  {avg_mae:.4f}")
    print("="*42)

    # ==========================================
    # 绘制并保存 PR 曲线图
    # ==========================================
    plt.figure(figsize=(8, 8))
    plt.plot(R, P, color='blue', lw=2, label=f'Improved Model (ODS F1={ods_f1:.3f})')
    
    # 画辅助线和标记最佳 ODS 点
    plt.plot(ods_r, ods_p, 'ro', markersize=8)
    plt.text(ods_r+0.02, ods_p+0.02, f'ODS', color='red', fontweight='bold')
    
    # 画 F1 等高线 (视觉优化，顶级论文标配)
    f_scores = np.linspace(0.1, 0.9, num=9)
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate(f'f={f_score:0.1f}', xy=(0.9, f_score * 0.9 / (2 * 0.9 - f_score)), alpha=0.5)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Precision-Recall Curve on BSDS500', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc="lower left")
    
    curve_path = "pr_curve.png"
    plt.savefig(curve_path, dpi=300, bbox_inches='tight')
    print(f"✅ PR 曲线已成功保存至当前目录: {curve_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 注意这里需要安装 matplotlib 库 (pip install matplotlib)
    parser.add_argument("--pred_dir", type=str, default="/home/sty/pyfile/sketchKeras_pytorch/test_results/improve", help="模型预测结果保存的文件夹")
    parser.add_argument("--gt_dir", type=str, default="data/test/masks", help="真实标签 (ground truth) 的文件夹")
    parser.add_argument("--num_thresh", type=int, default=99, help="用于遍历的阈值数量 (默认 99 步)")
    args = parser.parse_args()
    
    evaluate_dataset(args.pred_dir, args.gt_dir, args.num_thresh)