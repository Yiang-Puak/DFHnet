import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
import os

# 从我们修改好的模型文件中导入 DFHNet
from model import DFHNet


def calculate_metrics(y_true, y_pred_probs):
    """
    计算 Fmax 和 AUPR 指标。
    - y_true: 真实的标签矩阵 [num_samples, num_go_terms]
    - y_pred_probs: 模型预测的概率矩阵 [num_samples, num_go_terms]
    """
    y_true = y_true.numpy()
    y_pred_probs = y_pred_probs.numpy()

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
                if f1 > fmax:
                    fmax = f1

    # --- 计算 AUPR (Macro Average) ---
    aupr_scores = []
    for i in range(y_true.shape[1]):  # 遍历每个GO term
        if np.sum(y_true[:, i]) > 0:  # 只计算至少有一个正样本的类别
            precision, recall, _ = precision_recall_curve(y_true[:, i], y_pred_probs[:, i])
            aupr_scores.append(auc(recall, precision))

    macro_aupr = np.mean(aupr_scores) if aupr_scores else 0.0

    return fmax, macro_aupr


if __name__ == '__main__':
    print("--- 1. 开始加载模型和数据 ---")

    # --- 加载 GO 数据 ---
    try:
        go_data = torch.load('data/go_data.pth')
        go_adj_matrix = go_data['adj_matrix']
        GO_TERM_COUNT = len(go_data['go_to_idx'])
    except FileNotFoundError:
        print("错误: data/go_data.pth 未找到。请先运行 prepare_go_dag.py。")
        exit()

    # --- 加载模拟数据集 ---
    dataset_dir = "synthetic_dataset"
    train_path = os.path.join(dataset_dir, "train_dataset.pth")
    val_path = os.path.join(dataset_dir, "val_dataset.pth")

    try:
        train_dataset = torch.load(train_path)
        val_dataset = torch.load(val_path)
        print("模拟数据集加载成功。")
    except FileNotFoundError:
        print(f"错误: 在 '{dataset_dir}' 文件夹中未找到数据集。请先运行 create_and_save_dataset.py。")
        exit()

    # --- 模型参数 ---
    NUM_DOMAINS = 5000
    DOMAIN_EMBED_DIM = 256
    FEATURE_DIM = 1280
    PROTEIN_EMBED_DIM = 1024
    EPOCHS = 10

    # --- 实例化模型、损失函数和优化器 ---
    model = DFHNet(
        num_domains=NUM_DOMAINS, domain_embed_dim=DOMAIN_EMBED_DIM, go_term_count=GO_TERM_COUNT,
        agcn_in_dim=FEATURE_DIM, mcnn_in_dim=FEATURE_DIM, go_adj_matrix=go_adj_matrix,
        protein_embed_dim=PROTEIN_EMBED_DIM
    )
    # 推荐使用 FocalLoss，但为保持脚本简洁性，此处使用标准的 BCE Loss
    criterion = F.binary_cross_entropy
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # --- 2. 开始训练 ---
    print("\n--- 2. 开始训练 ---")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for sample in train_dataset:
            optimizer.zero_grad()
            probs, _ = model(sample["graph"], sample["domains"])
            loss = criterion(probs.squeeze(0), sample["labels"])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataset)
        alpha = torch.sigmoid(model.fusion.alpha).item()
        print(f"Epoch {epoch + 1}/{EPOCHS} | 平均损失: {avg_loss:.4f} | GCN权重(alpha): {alpha:.4f}")

    # --- 3. 开始验证和评估 ---
    print("\n--- 3. 开始验证和评估 ---")
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for sample in val_dataset:
            probs, _ = model(sample["graph"], sample["domains"])
            all_labels.append(sample["labels"])
            all_preds.append(probs.squeeze(0))

    all_labels_tensor = torch.stack(all_labels)
    all_preds_tensor = torch.stack(all_preds)

    fmax, macro_aupr = calculate_metrics(all_labels_tensor, all_preds_tensor)

    print("\n--- 4. 评估结果 ---")
    print(f"Fmax: {fmax:.4f}")
    print(f"Macro AUPR: {macro_aupr:.4f}")
    print("\n验证完成！")