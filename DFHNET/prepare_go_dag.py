# prepare_go_dag.py
import torch
import obonet
import networkx as nx
import numpy as np


def prepare_go_data(go_file_path='data/go.obo', output_path='data/go_data.pth'):
    """
    读取 go.obo 文件, 创建 GO term 的邻接矩阵和索引映射.
    """
    print("Loading GO graph from obo file...")
    graph = obonet.read_obo(go_file_path)

    # 筛选出三个主要分支: BP, MF, CC
    valid_namespaces = {'biological_process', 'molecular_function', 'cellular_component'}
    valid_nodes = [node for node, data in graph.nodes(data=True) if data.get('namespace') in valid_namespaces]

    # 创建一个只包含有效节点的子图
    go_dag = graph.subgraph(valid_nodes)

    # 获取所有 GO terms 并创建一个从 GO ID 到整数索引的映射
    go_terms = sorted(list(go_dag.nodes()))
    go_to_idx = {term: i for i, term in enumerate(go_terms)}
    num_go_terms = len(go_terms)
    print(f"Found {num_go_terms} valid GO terms.")

    # 创建邻接矩阵 A, A[i, j] = 1 表示 GO term j 是 GO term i 的父节点
    adj_matrix = np.zeros((num_go_terms, num_go_terms), dtype=np.float32)

    for term, idx in go_to_idx.items():
        # networkx中, successor 是子节点, predecessor 是父节点
        # 我们要找父节点,所以用 predecessors
        for parent in go_dag.predecessors(term):
            if parent in go_to_idx:
                parent_idx = go_to_idx[parent]
                # 在我们的定义中, adj_matrix[i,j]=1表示j是i的父节点
                adj_matrix[idx, parent_idx] = 1

    # 将Numpy矩阵转换为PyTorch Tensor
    adj_matrix_tensor = torch.from_numpy(adj_matrix)

    # 保存邻接矩阵和索引映射
    torch.save({
        'adj_matrix': adj_matrix_tensor,
        'go_to_idx': go_to_idx,
        'idx_to_go': {i: term for term, i in go_to_idx.items()}
    }, output_path)

    print(f"GO data saved to {output_path}")


if __name__ == '__main__':
    prepare_go_data()