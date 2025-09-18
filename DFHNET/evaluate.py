# evaluate.py
import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_recall_curve, auc

def calculate_metrics(y_true, y_pred_probs):
    """
    计算 Fmax 和 AUPR 指标。
    - y_true: 真实的标签矩阵 [num_samples, num_go_terms]
    - y_pred_probs: 模型预测的概率矩阵 [num_samples, num_go_terms]
    """
    # 将数据移回CPU并转换为numpy数组进行计算
    y_true = y_true.cpu().numpy()
    y_pred_probs = y_pred_probs.cpu().numpy()
    
    # --- 计算 Fmax ---
    fmax = 0.0
    for threshold in np.linspace(0, 1, 101):
        precision_total, recall_total = 0, 0
        num_proteins_with_preds = 0
        for i in range(y_true.shape[0]):
            preds = y_pred_probs[i] > threshold
            true_positives = np.sum(preds & y_true[i].astype(bool))
            if np.sum(preds) > 0:
                num_proteins_with_preds += 1
                precision_total += true_positives / np.sum(preds)
            if np.sum(y_true[i]) > 0:
                recall_total += true_positives / np.sum(y_true[i])
        if num_proteins_with_preds > 0:
            avg_precision = precision_total / num_proteins_with_preds
            avg_recall = recall_total / y_true.shape[0]
            if avg_precision + avg_recall > 0:
                f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
                if f1 > fmax: fmax = f1
    
    # --- 计算 AUPR (Macro Average) ---
    aupr_scores = []
    for i in range(y_true.shape[1]):
        if np.sum(y_true[:, i]) > 0:
            precision, recall, _ = precision_recall_curve(y_true[:, i], y_pred_probs[:, i])
            aupr_scores.append(auc(recall, precision))
    macro_aupr = np.mean(aupr_scores) if aupr_scores else 0.0
    
    return fmax, macro_aupr

def evaluate_model(model, data_loader, device):
    """
    在给定的数据集上评估模型。
    此版本已更新，可正确处理 PyTorch Geometric 的批次数据。
    """
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        eval_pbar = tqdm(data_loader, desc="[Validation]")
        for batch_data in eval_pbar:
            # 1. 将整个批次数据一次性移动到 GPU
            batch_data = batch_data.to(device)

            # 2. 将 Batch 对象分解为 Data 对象列表
            graph_list = batch_data.to_data_list()
            
            # 3. 逐个处理批次中的每个图
            for graph in graph_list:
                probs, _ = model(graph, graph.domains)
                all_preds.append(probs)
                all_labels.append(graph.labels)
            
    # 将所有结果和标签合并成一个大张量
    all_labels_tensor = torch.cat(all_labels, dim=0)
    all_preds_tensor = torch.cat(all_preds, dim=0)
    
    # 计算并返回指标
    return calculate_metrics(all_labels_tensor, all_preds_tensor)