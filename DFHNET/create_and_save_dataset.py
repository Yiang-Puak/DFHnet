# create_and_save_dataset.py (REVISED)
import torch
import numpy as np
from torch_geometric.data import Data
import os

def create_synthetic_dataset(num_samples, seq_len_range, feature_dim, go_term_count, num_domains):
    """
    创建一个包含多个样本的简单模拟数据集。
    新版：将 domains 和 labels 直接作为 Data 对象的属性。
    """
    dataset = []
    for i in range(num_samples):
        seq_length = np.random.randint(*seq_len_range)
        
        # 1. 创建基础图数据
        node_features = torch.randn(seq_length, feature_dim)
        num_edges = np.random.randint(seq_length, seq_length * 2)
        edge_index = torch.randint(0, seq_length, (2, num_edges))
        graph_data = Data(x=node_features, edge_index=edge_index)
        
        # 2. 创建域ID和标签
        num_protein_domains = np.random.randint(1, 6)
        domain_ids = torch.LongTensor(np.random.randint(0, num_domains, size=num_protein_domains))
        
        num_labels = np.random.randint(5, 21)
        labels = torch.zeros(1, go_term_count) # 保持 [1, C] 的形状
        label_indices = np.random.randint(0, go_term_count, size=num_labels)
        labels[0, label_indices] = 1.0
        
        # 3. 将 domains 和 labels 作为属性附加到 graph_data 对象上
        graph_data.domains = domain_ids
        graph_data.labels = labels
        
        dataset.append(graph_data)
        
    print(f"成功创建 {num_samples} 个 PyG Data 对象。")
    return dataset

if __name__ == '__main__':
    print("--- 1. 开始生成 PyG 格式的模拟数据集 ---")
    
    try:
        go_data = torch.load('data/go_data.pth', weights_only=False)
        GO_TERM_COUNT = len(go_data['go_to_idx'])
    except FileNotFoundError:
        print("错误: data/go_data.pth 未找到。请先运行 prepare_go_dag.py。")
        exit()

    NUM_DOMAINS = 5000
    FEATURE_DIM = 1280
    NUM_TRAIN_SAMPLES = 100
    NUM_VAL_SAMPLES = 20
    SEQ_LEN_RANGE = (100, 300)

    train_dataset = create_synthetic_dataset(NUM_TRAIN_SAMPLES, SEQ_LEN_RANGE, FEATURE_DIM, GO_TERM_COUNT, NUM_DOMAINS)
    val_dataset = create_synthetic_dataset(NUM_VAL_SAMPLES, SEQ_LEN_RANGE, FEATURE_DIM, GO_TERM_COUNT, NUM_DOMAINS)
    
    output_dir = "synthetic_dataset"
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, "train_dataset.pth")
    val_path = os.path.join(output_dir, "val_dataset.pth")
    
    # 保存 Data 对象列表
    torch.save(train_dataset, train_path)
    torch.save(val_dataset, val_path)
    
    print(f"\n--- 2. 数据集已成功保存 ---")
    print(f"训练集路径: {train_path}")
    print(f"验证集路径: {val_path}")