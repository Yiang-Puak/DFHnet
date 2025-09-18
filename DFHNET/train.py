# train.py
import torch
from evaluate import evaluate_model
from tqdm import tqdm

def train_model(model, train_loader, val_loader, optimizer, criterion, epochs, device):
    """
    模型训练的主循环。
    此版本已更新，可正确处理 PyTorch Geometric 的批次数据。
    """
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        # 使用 tqdm 显示带有描述的进度条
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Training]")
        for batch_data in train_pbar:
            # 1. 将整个批次数据一次性移动到 GPU
            batch_data = batch_data.to(device)
            optimizer.zero_grad()

            # 2. 将一个 Batch 对象分解成一个包含单个 Data 对象的 Python 列表
            graph_list = batch_data.to_data_list()
            
            # 用于存储批次内每个样本的输出和标签
            batch_probs = []
            batch_labels = []

            # 3. 逐个处理批次中的每个图（因为模型当前 forward 函数设计为处理单个图）
            for graph in graph_list:
                probs, _ = model(graph, graph.domains)
                batch_probs.append(probs)
                batch_labels.append(graph.labels)
            
            # 4. 将该批次所有样本的结果和标签重新合并成一个张量
            #    例如，如果 batch_size=8, final_probs 的维度将是 [8, num_classes]
            final_probs = torch.cat(batch_probs, dim=0)
            final_labels = torch.cat(batch_labels, dim=0)

            # 5. 计算损失，此时维度匹配，不会报错
            loss = criterion(final_probs, final_labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            # 在进度条后面显示当前批次的损失
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        avg_loss = total_loss / len(train_loader)
        alpha = torch.sigmoid(model.fusion.alpha).item()
        
        # --- 每个 epoch 结束后进行验证 ---
        fmax, macro_aupr = evaluate_model(model, val_loader, device)
        
        # 打印当前周期的完整信息
        print(f"\nEpoch {epoch + 1}/{epochs} | "
              f"训练损失: {avg_loss:.4f} | "
              f"GCN权重(alpha): {alpha:.4f} | "
              f"验证 Fmax: {fmax:.4f} | "
              f"验证 Macro AUPR: {macro_aupr:.4f}")