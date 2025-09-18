# main.py (REVISED)
import torch
import torch.optim as optim
import torch.nn.functional as F
import os

# 从其他模块导入必要的组件
from model import DFHNet
from train import train_model
from utils import FocalLoss

# ！！！关键改动 1: 导入 PyG 的 DataLoader 和 Dataset ！！！
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

class SyntheticDataset(Dataset):
    """
    加载 PyG Data 对象列表的 Dataset 类。
    """
    def __init__(self, data_path):
        super(SyntheticDataset, self).__init__()
        self.data = torch.load(data_path, weights_only=False)
        print(f"从 {data_path} 加载了 {len(self.data)} 个 PyG Data 对象。")
        
    def len(self):
        return len(self.data)
        
    def get(self, idx):
        return self.data[idx]

if __name__ == '__main__':
    # --- 1. 环境和参数设置 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- 使用设备: {device} ---")

    # --- 2. 加载 GO 和模拟数据集 ---
    try:
        go_data = torch.load('data/go_data.pth', weights_only=False)
        go_adj_matrix = go_data['adj_matrix'].to(device)
        GO_TERM_COUNT = len(go_data['go_to_idx'])
    except FileNotFoundError:
        print("错误: data/go_data.pth 未找到。请先运行 prepare_go_dag.py。")
        exit()

    dataset_dir = "synthetic_dataset"
    train_path = os.path.join(dataset_dir, "train_dataset.pth")
    val_path = os.path.join(dataset_dir, "val_dataset.pth")
    if not os.path.exists(train_path):
        print(f"错误: 在 '{dataset_dir}' 文件夹中未找到数据集。请先运行 create_and_save_dataset.py。")
        exit()
        
    train_dataset = SyntheticDataset(train_path)
    val_dataset = SyntheticDataset(val_path)
    
    # ！！！关键改动 2: 使用 PyG 的 DataLoader ！！！
    # batch_size > 1 现在也可以正常工作了
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # --- 3. 实例化模型、损失函数和优化器 ---
    model_params = {
        'num_domains': 5000, 'domain_embed_dim': 256, 'go_term_count': GO_TERM_COUNT,
        'agcn_in_dim': 1280, 'mcnn_in_dim': 1280, 'go_adj_matrix': go_adj_matrix,
        'protein_embed_dim': 1024
    }
    model = DFHNet(**model_params).to(device)
    
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    print(f"使用 {type(criterion).__name__} 作为损失函数。")
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    EPOCHS = 10
    
    # --- 4. 启动训练 (需要修改 train.py 来适应新的数据格式) ---
    print("\n--- 开始启动训练流程 ---")
    # 注意：您还需要对 train.py 和 evaluate.py 做微小调整
    train_model(
        model=model, train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer, criterion=criterion, epochs=EPOCHS, device=device
    )
    
    print("\n--- 训练完成！ ---")