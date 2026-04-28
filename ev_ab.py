import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

# ==========================================
# 解决 OpenCV 中文路径读写问题的辅助函数
# ==========================================
def cv_imread_chinese_gray(file_path):
    """支持中文路径的灰度图读取"""
    return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

# ==========================================
# 核心指标计算逻辑
# ==========================================
def evaluate_single_model(pred_dir, gt_dir, num_thresh=99, tolerance=2):
    """
    评估单个模型的预测结果，返回完整的指标字典
    """
    valid_exts = ['.png', '.jpg', '.jpeg', '.bmp']
    pred_files = [f for f in os.listdir(pred_dir) if os.path.splitext(f)[1].lower() in valid_exts]

    if not pred_files:
        return None

    # 生成 0.01 到 0.99 的阈值数组 (用于寻找最佳阈值)
    thresholds = np.linspace(0.01, 0.99, num_thresh)

    dataset_tp_pred = np.zeros(num_thresh)
    dataset_tp_gt = np.zeros(num_thresh)
    dataset_pred_sum = np.zeros(num_thresh)
    dataset_gt_sum = np.zeros(num_thresh)

    total_mae = 0.0
    img_best_f1s = []
    valid_count = 0

    for pred_file in tqdm(pred_files, desc=f"  评估模型: {os.path.basename(pred_dir)}", leave=False):
        pred_path = os.path.join(pred_dir, pred_file)
        # 统一使用 .png 去真值文件夹找对应的掩码
        gt_path = os.path.join(gt_dir, os.path.splitext(pred_file)[0] + '.png')

        if not os.path.exists(gt_path):
            continue

        # 使用支持中文路径的读取方式
        pred_img = cv_imread_chinese_gray(pred_path)
        pred_prob = pred_img.astype(np.float32) / 255.0

        gt_img = cv_imread_chinese_gray(gt_path)

        # 尺寸对齐保护
        if pred_img.shape != gt_img.shape:
            pred_prob = cv2.resize(pred_prob, (gt_img.shape[1], gt_img.shape[0]))

        # 1. 计算 MAE (平均绝对误差)
        gt_norm = (gt_img > 0).astype(np.float32)
        total_mae += np.mean(np.abs(pred_prob - gt_norm))

        # 2. 准备边缘容差 (Tolerance) 膨胀操作，解决人工标注的轻微偏移问题
        gt_binary = (gt_img > 0).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tolerance*2+1, tolerance*2+1))
        gt_dilated = cv2.dilate(gt_binary, kernel)

        img_f1_list = np.zeros(num_thresh)

        # 遍历所有阈值，计算 PR 和 F1
        for i, t in enumerate(thresholds):
            pred_binary = (pred_prob >= t).astype(np.uint8)
            pred_dilated = cv2.dilate(pred_binary, kernel)

            # 计算 True Positives
            tp_pred = np.sum(np.logical_and(pred_binary == 1, gt_dilated == 1))
            tp_gt = np.sum(np.logical_and(gt_binary == 1, pred_dilated == 1))
            sum_p = np.sum(pred_binary)
            sum_g = np.sum(gt_binary)

            dataset_tp_pred[i] += tp_pred
            dataset_tp_gt[i] += tp_gt
            dataset_pred_sum[i] += sum_p
            dataset_gt_sum[i] += sum_g

            # 当前单张图片的 Precision 和 Recall
            p = tp_pred / sum_p if sum_p > 0 else 0
            r = tp_gt / sum_g if sum_g > 0 else 0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            img_f1_list[i] = f1

        # 记录单张图片在最优阈值下的 F1 (用于计算 OIS)
        img_best_f1s.append(np.max(img_f1_list))
        valid_count += 1

    if valid_count == 0:
        return None

    # 计算整个数据集级别的 P, R, F1 (用于计算 ODS)
    P = np.divide(dataset_tp_pred, dataset_pred_sum, out=np.zeros_like(dataset_tp_pred), where=dataset_pred_sum!=0)
    R = np.divide(dataset_tp_gt, dataset_gt_sum, out=np.zeros_like(dataset_tp_gt), where=dataset_gt_sum!=0)
    F1 = np.divide(2 * P * R, P + R, out=np.zeros_like(P), where=(P+R)!=0)

    ods_index = np.argmax(F1) # 找到使得整个数据集 F1 最高的全局最优阈值

    return {
        'ods_thresh': thresholds[ods_index],
        'ods_f1': F1[ods_index],
        'ois_f1': np.mean(img_best_f1s),
        'ods_p': P[ods_index],
        'ods_r': R[ods_index],
        'mae': total_mae / valid_count,
        'P_array': P,
        'R_array': R
    }

# ==========================================
# 场景评估与可视化
# ==========================================
def evaluate_scenario(base_dir, gt_dir, scenario_name, num_thresh=99, tolerance=2):
    """
    评估某个大场景下的所有模型
    """
    print(f"\n" + "="*80)
    print(f"🚀 开始评估场景: {scenario_name}".center(75))
    print("="*80)

    if not os.path.exists(base_dir):
        print(f"❌ 找不到文件夹: {base_dir}，请检查路径。")
        return

    # 遍历当前场景下的所有模型预测文件夹
    model_names = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])

    all_results = {}
    for model_name in model_names:
        pred_dir = os.path.join(base_dir, model_name)
        res = evaluate_single_model(pred_dir, gt_dir, num_thresh, tolerance)
        if res is not None:
            all_results[model_name] = res

    if not all_results:
        print(f"⚠️ 场景 {scenario_name} 下未找到任何有效预测结果。")
        return

    # 打印排版好的大表格
    print(f"\n📊 {scenario_name} 综合排名表")
    print("-" * 80)
    print(f"{'Model Name':<20} | {'ODS F1':<8} | {'OIS F1':<8} | {'Precision':<9} | {'Recall':<8} | {'MAE':<8}")
    print("-" * 80)

    # 按 ODS F1 降序排列
    sorted_models = sorted(all_results.items(), key=lambda x: x[1]['ods_f1'], reverse=True)
    for model_name, metrics in sorted_models:
        print(f"{model_name:<20} | {metrics['ods_f1']:.4f}   | {metrics['ois_f1']:.4f}   | {metrics['ods_p']:.4f}    | {metrics['ods_r']:.4f}   | {metrics['mae']:.4f}")
    print("-" * 80)

    # 绘制 PR 曲线
    plt.figure(figsize=(10, 8))
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'cyan', 'magenta', 'lime', 'navy']

    for idx, (model_name, metrics) in enumerate(sorted_models):
        color = colors[idx % len(colors)]
        label_name = f"{model_name} (ODS={metrics['ods_f1']:.3f})"

        # 突出您的最终形态模型 (加粗线条)
        lw = 3.0 if "Exp4" in model_name or "MNet_Full" in model_name else 1.5

        plt.plot(metrics['R_array'], metrics['P_array'], color=color, lw=lw, label=label_name)
        plt.plot(metrics['ods_r'], metrics['ods_p'], marker='o', color=color, markersize=6)

    # 画 F1 等高线
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
    plt.title(f'Precision-Recall Curves ({scenario_name})', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc="lower left", fontsize=10)

    # 兼容中文路径保存图片
    curve_path = f"PR_Curve_{scenario_name}.png"
    plt.savefig(curve_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 该场景 PR 曲线已保存至项目根目录: {curve_path}")

if __name__ == "__main__":
    # 您统一的真值标签(Ground Truth) 目录
    GT_DIR = r"C:\Users\酸雨\Desktop\毕设—边缘检测\sketchKeras_pytorch\data\test\masks"

    # 消融实验的结果目录
    ABLATION_RES_DIR = r"C:\Users\酸雨\Desktop\毕设—边缘检测\sketchKeras_pytorch\test_results_ablation"

    # 您需要评估的具体场景 (对应您在 ab_test_all.py 中生成的三大文件夹)
    scenarios = ["Original", "Gaussian", "SaltPepper"]

    print("🎯 初始化边缘检测全量指标评估引擎...")

    for scenario in scenarios:
        scenario_path = os.path.join(ABLATION_RES_DIR, scenario)
        evaluate_scenario(scenario_path, GT_DIR, scenario_name=scenario)

    print("\n🎉 全部评估大功告成！PR曲线图已生成，可以直接复制表格数据粘贴到您的论文里了！")