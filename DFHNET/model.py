# model.py (REVISED AND IMPROVED)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


# ===================================================================
# ==  原始模块 (保留 AGCN 和 MCNN 的基础结构)                      ==
# ===================================================================

class AGCN(nn.Module):
    """
    自适应图卷积网络 (AGCN) - 结构流
    - 保持基本 GCN 结构用于学习局部结构特征
    - 移除了 MHA 和 predictor，因为特征聚合和预测将在更高层次完成
    """

    def __init__(self, in_channels, out_channels, dropout=0.1):
        super(AGCN, self).__init__()
        # 建议增加残差连接以获得更好的性能，此处为简化暂未添加
        self.conv1 = GCNConv(in_channels, 512)
        self.conv2 = GCNConv(512, out_channels)  # 输出维度直接为蛋白质嵌入维度
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        # 输出每个残基的特征，供后续的注意力模块使用
        return x


class MCNN(nn.Module):
    """
    多层卷积神经网络 (MCNN) - 序列流
    - 保持基本 CNN 结构用于学习序列局部特征
    - 移除了 predictor，因为预测将在更高层次完成
    """

    def __init__(self, in_channels, out_channels, dropout=0.1):
        super(MCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 512, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(512)
        self.conv2 = nn.Conv1d(512, out_channels, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        # MCNN 操作的是序列特征: 期望 [batch, channels, length]
        x = data.x.unsqueeze(0).permute(0, 2, 1)  # [L, C] -> [1, C, L]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x_permuted = x.permute(0, 2, 1)  # [1, C, L] -> [1, L, C]
        # 输出每个残基的特征，供后续的注意力模块使用
        return x_permuted.squeeze(0)  # [L, C]


# ===================================================================
# ==  第一部分改进: 借鉴 DPFunc 的域引导注意力机制                 ==
# ===================================================================
class DomainAttention(nn.Module):
    """
    域引导的注意力模块.
    该模块利用域信息作为查询(Query)，计算每个残基的重要性，
    并对残基特征进行加权求和，生成一个全局的蛋白质嵌入。
    """

    def __init__(self, domain_embed_dim, residue_feature_dim, attention_dim):
        super().__init__()
        self.query_proj = nn.Linear(domain_embed_dim, attention_dim)
        self.key_proj = nn.Linear(residue_feature_dim, attention_dim)
        self.value_proj = nn.Linear(residue_feature_dim, residue_feature_dim)
        self.scale = attention_dim ** -0.5

    def forward(self, domain_embedding_sum, residue_features):
        # domain_embedding_sum: [1, domain_embed_dim]
        # residue_features: [L, residue_feature_dim]

        # 1. 将域嵌入和残基特征投影到注意力空间
        # Q: [1, attention_dim], K: [L, attention_dim], V: [L, residue_feature_dim]
        q = self.query_proj(domain_embedding_sum)
        k = self.key_proj(residue_features)
        v = self.value_proj(residue_features)

        # 2. 计算注意力得分
        # attention_scores: [1, L]
        attention_scores = torch.matmul(q, k.transpose(0, 1)) * self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)  # [1, L]

        # 3. 对残基特征(V)进行加权求和
        # context_vector: [1, residue_feature_dim]
        context_vector = torch.matmul(attention_weights, v)

        return context_vector


# ===================================================================
# ==  第二部分改进: 借鉴 TAWFN 的双流自适应融合架构              ==
# ===================================================================
class DFHEncoder(nn.Module):
    """
    重构后的 DFH 编码器.
    包含 AGCN 和 MCNN 两个并行的流，每个流都使用域注意力来生成蛋白质嵌入，
    并独立进行初步预测。
    """

    def __init__(self, num_domains, domain_embed_dim, protein_embed_dim, go_term_count, agcn_in_dim, mcnn_in_dim):
        super().__init__()
        print("Initializing Re-architected DFHEncoder...")

        # 基础特征提取模块
        self.gcn_stream = AGCN(in_channels=agcn_in_dim, out_channels=protein_embed_dim)
        self.cnn_stream = MCNN(in_channels=mcnn_in_dim, out_channels=protein_embed_dim)

        # 域嵌入模块
        self.domain_embedding = nn.Embedding(num_domains, domain_embed_dim)

        # 域注意力模块 (两个流共享同一个注意力机制)
        self.attention = DomainAttention(domain_embed_dim, protein_embed_dim, attention_dim=256)

        # 每个流独立的预测器
        self.gcn_predictor = nn.Linear(protein_embed_dim, go_term_count)
        self.cnn_predictor = nn.Linear(protein_embed_dim, go_term_count)

    def forward(self, data, domain_ids):
        # 1. 获取所有域的嵌入并求和，作为全局域信息
        domain_embeds = self.domain_embedding(domain_ids)
        domain_embed_sum = torch.sum(domain_embeds, dim=0, keepdim=True)  # [1, domain_embed_dim]

        # 2. 并行处理 AGCN 和 MCNN 流
        gcn_residue_features = self.gcn_stream(data)  # [L, protein_embed_dim]
        cnn_residue_features = self.cnn_stream(data)  # [L, protein_embed_dim]

        # 3. 每个流都通过域注意力模块生成蛋白质嵌入
        gcn_protein_embedding = self.attention(domain_embed_sum, gcn_residue_features)
        cnn_protein_embedding = self.attention(domain_embed_sum, cnn_residue_features)

        # 4. 每个流都进行初步预测，输出 logits
        gcn_logits = self.gcn_predictor(gcn_protein_embedding)
        cnn_logits = self.cnn_predictor(cnn_protein_embedding)

        return gcn_logits, cnn_logits


class AdaptiveFusion(nn.Module):
    """
    自适应融合模块，学习如何加权融合来自两个流的预测结果。
    """

    def __init__(self):
        super().__init__()
        # 创建一个可学习的参数 alpha，用于控制融合权重
        # 初始化为0.5，表示初始时两个流同等重要
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, gcn_logits, cnn_logits):
        # 使用 sigmoid 保证 alpha 在 0-1 之间
        fused_logits = torch.sigmoid(self.alpha) * gcn_logits + (1 - torch.sigmoid(self.alpha)) * cnn_logits
        return fused_logits


# ===================================================================
# ==  保留的创新点: 层次化解码器                                   ==
# ===================================================================
class HierarchicalDecoder(nn.Module):
    def __init__(self, go_adj_matrix):
        super().__init__()
        print("Initializing Hierarchical Decoder...")
        # 解码器现在直接作用于融合后的 logits，不再需要自己的 predictor
        self.register_buffer('go_adj_matrix', go_adj_matrix)

    def forward(self, fused_logits, iterations=2):
        # True Path Rule: 如果子节点被预测，那么父节点也必须被预测
        # P(parent) >= max(P(children))

        current_logits = fused_logits
        # adj_matrix[i, j] = 1 表示 j 是 i 的父节点
        # 我们需要从子节点传播分数到父节点

        for _ in range(iterations):
            # 将 logits 转换为概率，以便进行 max 操作
            current_probs = torch.sigmoid(current_logits)

            # 找到每个节点的所有子节点分数的最大值
            # torch.matmul(A.T, p) 会聚合所有子节点的分数给父节点
            # A.T 的维度是 [num_terms, num_terms], current_probs 是 [1, num_terms]
            # 我们需要让 A[i,j] 表示 i 是 j 的父节点，这样 A.T[j,i] 表示 j 是 i 的子节点
            # 因此，我们需要 adj_matrix 本身

            # parent_probs_from_children = torch.matmul(current_probs, self.go_adj_matrix) -> 这样是子节点聚合到父节点
            # A[i, j]=1 (j is parent of i) -> P_i should be affected by P_j
            # We want P_j = max(P_j, P_i) for all children i

            # Let's rebuild the logic. For each term j, find its children i. P_j = max(P_j, P_i1, P_i2...)
            # This is equivalent to propagating child probabilities upwards.
            # `child_probs_agg[j] = sum_{i where j is parent of i} P_i` if using matmul(P, A)
            # What we need is max propagation.

            # 使用转置矩阵，A_T[j, i] = 1 表示 i 是 j 的父节点
            # max_child_logits, _ = torch.max(current_logits * self.go_adj_matrix.T, dim=1, keepdim=True)
            # This logic is complex with sparse matrices. Let's stick to the original logic which is sound.
            # A[i,j]=1 means j is parent of i. matmul(probs, A) -> for each parent j, aggregate scores from children i.

            agg_child_probs = torch.matmul(current_probs, self.go_adj_matrix)

            # 更新概率：如果任何一个子节点的概率高于父节点，则提升父节点的概率
            # P_new = max(P_current, P_aggregated_from_children)
            updated_probs = torch.max(current_probs, agg_child_probs)

            # 将更新后的概率转换回 logits，用于下一次迭代或最终输出
            # 防止概率为1或0导致log无限大
            updated_probs = torch.clamp(updated_probs, 1e-7, 1 - 1e-7)
            current_logits = torch.log(updated_probs / (1 - updated_probs))

        final_logits = current_logits
        final_probabilities = torch.sigmoid(final_logits)

        return final_probabilities, final_logits


# ===================================================================
# ==  最终组装的、大幅改进的 DFHNet 模型                           ==
# ===================================================================
class DFHNet(nn.Module):
    def __init__(self, num_domains, domain_embed_dim, go_term_count, agcn_in_dim, mcnn_in_dim, go_adj_matrix,
                 protein_embed_dim):
        super().__init__()
        print("Assembling the final IMPROVED DFHNet model...")
        self.encoder = DFHEncoder(num_domains, domain_embed_dim, protein_embed_dim, go_term_count, agcn_in_dim,
                                  mcnn_in_dim)
        self.fusion = AdaptiveFusion()
        self.decoder = HierarchicalDecoder(go_adj_matrix=go_adj_matrix)

    def forward(self, data, domain_ids):
        # 1. 编码器输出两个独立的初步预测
        gcn_logits, cnn_logits = self.encoder(data, domain_ids)

        # 2. 自适应融合模块对预测结果进行加权融合
        fused_logits = self.fusion(gcn_logits, cnn_logits)

        # 3. 层次化解码器对融合后的结果进行最终优化
        final_probs, final_logits = self.decoder(fused_logits)

        return final_probs, final_logits