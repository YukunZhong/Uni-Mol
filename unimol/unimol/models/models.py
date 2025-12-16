import torch
import torch.nn.functional as F
from torch_geometric.nn import GINEConv,GCNConv, GINConv, MessagePassing, global_mean_pool

# 1. UniMolGNN 空间高斯核 + 双通道更新
# === Gaussian Distance Encoder ===
# 将标量距离映射为多尺度的径向基函数 (RBF) 特征
# d: 张量，形状为 (num_edges,)，表示每条边的欧式距离
# 返回: 形状 (num_edges, num_kernels) 的张量，每行是对应距离的 RBF 响应
def gaussian_distance_encoder(d, mu_min=0.0, mu_max=5.0, num_kernels=32):
    # 在 [mu_min, mu_max] 区间均匀采样 num_kernels 个中心
    mu = torch.linspace(mu_min, mu_max, num_kernels, device=d.device)
    # 带宽 beta 定义为中心之间的间隔。
    beta = (mu_max - mu_min) / (num_kernels - 1)
    # 将 d 从 (num_edges,) 扩展为 (num_edges, 1)
    d = d.unsqueeze(-1)
    # 计算每个距离与每个中心的高斯响应: exp(- (d - mu)^2 / beta)
    # 输出 (num_edges, num_kernels)
    return torch.exp(-((d - mu) ** 2) / beta)


# === Dual Channel GNN Layer ===
# 同时维护节点 (Atom) 特征 x 和边 (Pair) 特征 edge_attr
class DualChannelLayer(MessagePassing):
    def __init__(self, node_dim, edge_dim):
        # 使用 "add" 聚合邻居消息
        super().__init__(aggr='add')
        # Atom -> Pair: 更新边特征的 MLP
        # 输入: 拼接了 (h_i, h_j, edge_attr) 三者，总维度 = 2*node_dim + edge_dim
        # 输出: edge_dim
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * node_dim + edge_dim, edge_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(edge_dim, edge_dim)
        )
        # 将更新后的边特征映射到节点特征维度，用作注意力中的偏置
        self.edge_proj = torch.nn.Linear(edge_dim, node_dim)
        # 注意力打分 MLP: 输入 (x_i, x_j, bias) 三者拼接，总维度 = 3*node_dim
        # 输出一个标量 [0,1] 作为注意力权重
        self.attn = torch.nn.Sequential(
            torch.nn.Linear(3 * node_dim, 1),
            torch.nn.Sigmoid()
        )
        # 节点更新后 MLP，将聚合结果与原始特征拼接
        # 输入维度 = 2 * node_dim, 输出 = node_dim
        self.node_mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * node_dim, node_dim),
            torch.nn.ReLU()
        )

    def forward(self, x, edge_index, edge_attr):
        # edge_index: shape (2, num_edges)，表示边的起点和终点节点索引
        # x:          shape (num_nodes, node_dim)
        # edge_attr: shape (num_edges, edge_dim)

        # --- Atom -> Pair 更新边特征 ---
        row, col = edge_index         # row: 源节点 idx, col: 目标节点 idx
        h_i = x[row]                  # 每条边的源节点特征
        h_j = x[col]                  # 每条边的目标节点特征
        # 拼接源节点、目标节点和当前边特征
        pair_input = torch.cat([h_i, h_j, edge_attr], dim=-1)
        # 用 MLP 更新后加残差
        edge_attr = edge_attr + self.edge_mlp(pair_input)

        # --- Message Passing: Node <- Neighbors ---
        # propagate 会调用 message() 来计算每条边的消息，并在目标节点聚合
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        # --- Pair -> Atom 更新节点特征 ---
        # 将原始节点特征与聚合消息拼接后通过 MLP
        x_updated = self.node_mlp(torch.cat([x, out], dim=-1))
        # 返回更新后的节点特征和边特征
        return x_updated, edge_attr

    def message(self, x_i, x_j, edge_attr):
        # message 在每条边上被调用，其中:
        # x_i: 目标节点特征 (dest)
        # x_j: 源节点特征 (src)
        # edge_attr: 对应边特征

        # --- 计算注意力偏置 ---
        bias = self.edge_proj(edge_attr)  # 映射到 node_dim
        # 将 x_i, x_j, bias 拼成一行，送入 attn MLP 得到注意力分数
        score = self.attn(torch.cat([x_i, x_j, bias], dim=-1))  # [0,1]
        # 返回消息: score * (邻居特征 + 偏置)
        return score * (x_j + bias)


# 将以上 RBF + DualChannelLayer 组合，构建端到端分子属性预测网络
class UniMolGNN(torch.nn.Module):
    def __init__(self, node_dim=13, edge_dim=32, hidden_dim=128, output_dim=1, num_layers=3):
        super().__init__()
        # 节点特征编码器: 将原子类型 one-hot 或其他特征映射到 hidden_dim
        self.node_encoder = torch.nn.Linear(node_dim, hidden_dim)
        # 边特征编码器: 将 RBF 特征 (32 维) 投影到 edge_dim
        self.edge_encoder = torch.nn.Linear(32, edge_dim)
        # 重复堆叠 DualChannelLayer
        self.layers = torch.nn.ModuleList([
            DualChannelLayer(hidden_dim, edge_dim)
            for _ in range(num_layers)
        ])
        # 全局预测头: 先 pooling 再 MLP
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, data):
        # data.x: shape (num_nodes, node_dim)
        # data.edge_dist: shape (num_edges,) 存储每条边的欧氏距离
        # data.edge_index: shape (2, num_edges)
        # data.batch: shape (num_nodes,) 标识哪条节点属于哪个分子

        # 1. 编码节点特征
        x = self.node_encoder(data.x)  # (num_nodes, hidden_dim)
        
        # 2. 编码边特征: RBF -> 线性投影
        edge_rbf = gaussian_distance_encoder(data.edge_dist)  # (num_edges, 32)
        edge_attr = self.edge_encoder(edge_rbf)               # (num_edges, edge_dim)

        # 3. 多层消息传递
        for layer in self.layers:
            x_res = x  # 保存残差连接的输入
            # Dual-channel 更新节点和边
            x, edge_attr = layer(x, data.edge_index, edge_attr)
            # 节点残差连接
            x = x + x_res

        # 4. 全局读出: 平均池化
        x = global_mean_pool(x, data.batch)  # (batch_size, hidden_dim)
        # 5. 预测
        return self.decoder(x)  # (batch_size, output_dim)


# 2. Edge-Enhanced GNN（使用简单边特征）
class EdgeEnhancedGNN(torch.nn.Module):
    def __init__(self, node_dim=13, edge_dim=1, hidden_dim=128, output_dim=1):
        super().__init__()
        self.node_encoder = torch.nn.Linear(node_dim, hidden_dim)
        self.edge_encoder = torch.nn.Linear(edge_dim, hidden_dim)
        
        self.layers = torch.nn.ModuleList([
            GINEConv(
                torch.nn.Sequential(
                    torch.nn.Linear(hidden_dim, hidden_dim),
                    torch.nn.ReLU()
                ),
                train_eps = True
            ) for _ in range(3)
        ])
        
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim//2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim//2, output_dim)
        )

    def forward(self, data):
        x = self.node_encoder(data.x)
        edge_attr = self.edge_encoder(data.edge_dist.unsqueeze(-1))
        
        for conv in self.layers:
            x = conv(x, data.edge_index,edge_attr) + x  # 残差连接
        x = global_mean_pool(x, data.batch)
        return self.decoder(x)

# 3. Basic GNN（忽略边特征）
class BasicGNN(torch.nn.Module):
    def __init__(self, node_dim=13, hidden_dim=128, output_dim=1):
        super().__init__()
        self.conv1 = GCNConv(node_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim//2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim//2, output_dim)
        )

    def forward(self, data):
        x = self.conv1(data.x, data.edge_index).relu()
        x = self.conv2(x, data.edge_index).relu()
        x = self.conv3(x, data.edge_index)
        x = global_mean_pool(x, data.batch)
        return self.decoder(x)

# 4. Enhanced MLP
class EnhancedMLP(torch.nn.Module):
    def __init__(self, input_dim=16, hidden_dim=128, output_dim=1):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, data):
        # 拼接原子特征和坐标 [node_dim(6) + 3]
        node_features = torch.cat([data.x, data.pos], dim=-1)
        # 全局平均池化
        graph_features = global_mean_pool(node_features, data.batch)
        return self.mlp(graph_features)