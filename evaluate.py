import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def evaluate_single_model(pred_dir, gt_dir, num_thresh=99, tolerance=2):
    """
    评估单个模型的预测结果，返回完整的指标字典
    """
    pred_files = [f for f in os.listdir(pred_dir) if f.endswith(('.png', '.jpg'))]
    
    if not pred_files:
        return None

    # 生成 0.01 到 0.99 的阈值数组
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
        gt_path = os.path.join(gt_dir, os.path.splitext(pred_file)[0] + '.png')
        
        if not os.path.exists(gt_path):
            continue
            
        pred_img = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        pred_prob = pred_img.astype(np.float32) / 255.0
        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        
        if pred_img.shape != gt_img.shape:
            pred_prob = cv2.resize(pred_prob, (gt_img.shape[1], gt_img.shape[0]))
            
        gt_norm = (gt_img > 0).astype(np.float32)
        total_mae += np.mean(np.abs(pred_prob - gt_norm))
        
        gt_binary = (gt_img > 0).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tolerance*2+1, tolerance*2+1))
        gt_dilated = cv2.dilate(gt_binary, kernel)
        
        img_f1_list = np.zeros(num_thresh)

        for i, t in enumerate(thresholds):
            pred_binary = (pred_prob >= t).astype(np.uint8)
            pred_dilated = cv2.dilate(pred_binary, kernel)
            
            tp_pred = np.sum(np.logical_and(pred_binary == 1, gt_dilated == 1))
            tp_gt = np.sum(np.logical_and(gt_binary == 1, pred_dilated == 1))
            sum_p = np.sum(pred_binary)
            sum_g = np.sum(gt_binary)
            
            dataset_tp_pred[i] += tp_pred
            dataset_tp_gt[i] += tp_gt
            dataset_pred_sum[i] += sum_p
            dataset_gt_sum[i] += sum_g
            
            p = tp_pred / sum_p if sum_p > 0 else 0
            r = tp_gt / sum_g if sum_g > 0 else 0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            img_f1_list[i] = f1
            
        img_best_f1s.append(np.max(img_f1_list))
        valid_count += 1
        
    if valid_count == 0:
        return None

    P = np.divide(dataset_tp_pred, dataset_pred_sum, out=np.zeros_like(dataset_tp_pred), where=dataset_pred_sum!=0)
    R = np.divide(dataset_tp_gt, dataset_gt_sum, out=np.zeros_like(dataset_tp_gt), where=dataset_gt_sum!=0)
    F1 = np.divide(2 * P * R, P + R, out=np.zeros_like(P), where=(P+R)!=0)
    
    ods_index = np.argmax(F1)
    
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

def evaluate_scenario(base_dir, gt_dir, num_thresh=99, tolerance=2):
    """
    评估某个大场景（如原图、高斯噪声）下的所有模型
    """
    scenario_name = os.path.basename(base_dir)
    print(f"\n" + "="*80)
    print(f"🚀 开始评估场景: {scenario_name}".center(75))
    print("="*80)
    
    if not os.path.exists(base_dir):
        print(f"❌ 找不到文件夹: {base_dir}，请检查路径。")
        return

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
    
    sorted_models = sorted(all_results.items(), key=lambda x: x[1]['ods_f1'], reverse=True)
    for model_name, metrics in sorted_models:
        print(f"{model_name:<20} | {metrics['ods_f1']:.4f}   | {metrics['ois_f1']:.4f}   | {metrics['ods_p']:.4f}    | {metrics['ods_r']:.4f}   | {metrics['mae']:.4f}")
    print("-" * 80)

    # 绘制 PR 曲线
    plt.figure(figsize=(10, 8))
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'cyan', 'magenta']
    
    for idx, (model_name, metrics) in enumerate(sorted_models):
        color = colors[idx % len(colors)]
        label_name = f"{model_name} (ODS={metrics['ods_f1']:.3f})"
        lw = 3.0 if "improve" in model_name.lower() else 1.5
        
        plt.plot(metrics['R_array'], metrics['P_array'], color=color, lw=lw, label=label_name)
        plt.plot(metrics['ods_r'], metrics['ods_p'], marker='o', color=color, markersize=6)

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
    
    curve_path = f"pr_curve_{scenario_name}.png"
    plt.savefig(curve_path, dpi=300, bbox_inches='tight')
    plt.close() # 关掉画板，防止多张图重叠
    print(f"✅ 该场景曲线已保存: {curve_path}")

if __name__ == "__main__":
    # ===============================
    # 在这里硬编码您的根目录路径
    # ===============================
    ROOT_DIR = "/home/sty/pyfile/sketchKeras_pytorch"
    GT_DIR = os.path.join(ROOT_DIR, "data/test/masks")
    
    # 根据您截图中的三大场景文件夹名称
    scenarios = [
        "test_results_origin",
        "test_results_gaussian",
        "test_results_salt_pepper"
    ]
    
    print("🎯 开始一键批量处理所有场景...")
    for scenario_folder in scenarios:
        scenario_path = os.path.join(ROOT_DIR, scenario_folder)
        evaluate_scenario(scenario_path, GT_DIR)
        
    print("\n🎉 全部大功告成！您可以去项目目录下查看那 3 张 PR 曲线图了！")