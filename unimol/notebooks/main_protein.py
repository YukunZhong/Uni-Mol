import os
import numpy as np
import joblib
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from models import EdgeEnhancedGNN, BasicGNN, EnhancedMLP, UniMolGNN
torch.serialization.add_safe_globals({"Data": Data})
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix

# 蛋白质原子类型映射（扩展常见元素）
ATOM_TYPES = {
    'H': 0,
    'C': 1,
    'N': 2,
    'O': 3,
    'S': 4,
    'P': 5,
    'FE': 6,
    'ZN': 7,
    'CL': 8,
    'F': 9,
    'BR': 10,
    'I': 11,
    'OTHER': 12  # 其他未知元素
}

class NRDLDDataset(Dataset):
    def __init__(self, root, pdb_dir, threshold=5.0, transform=None, pre_transform=None):
        self.root = root
        self.pdb_dir = pdb_dir
        self.threshold = threshold
        self.raw_data = []
        self.labels = []

        # 读取nrdld.txt
        with open(os.path.join(root, 'nrdld.txt'), 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 3:
                    continue
                pdb_code, category, _ = parts  # 忽略set字段
                self.raw_data.append(pdb_code)
                self.labels.append(1 if category == 'd' else 0)

        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return [f"{code}.pdb" for code in self.raw_data]

    @property
    def processed_file_names(self):
        return [f'data_{i}.pt' for i in range(len(self.raw_data))]

    def process(self):
        if len(os.listdir(self.processed_dir)) == len(self.raw_data):
            return

        for idx, pdb_code in enumerate(self.raw_data):
            pdb_path = os.path.join(self.pdb_dir, f"{pdb_code}.pdb")
            if not os.path.exists(pdb_path):
                continue

            # 解析PDB文件
            atoms = self._parse_pdb(pdb_path)

            print(pdb_code)
            
            # 生成节点特征和坐标
            node_features = []
            pos = []
            for element, coord in atoms:
                type_idx = ATOM_TYPES.get(element, ATOM_TYPES['OTHER'])
                one_hot = F.one_hot(torch.tensor(type_idx), num_classes=len(ATOM_TYPES)).float()
                node_features.append(one_hot)
                pos.append(coord)
            
            print("原子数 ",len(node_features))
            
            if not node_features:
                continue

            x = torch.stack(node_features)
            pos = torch.tensor(pos, dtype=torch.float)
            
            # 生成边
            edge_index, edge_dist = self._create_edges(pos)
            
            # 目标标签
            y = torch.tensor([self.labels[idx]], dtype=torch.float)
            
            data = Data(x=x, pos=pos, edge_index=edge_index, edge_dist=edge_dist, y=y)
            
            if self.pre_filter and not self.pre_filter(data):
                continue
                
            if self.pre_transform:
                data = self.pre_transform(data)
                
            torch.save(data, os.path.join(self.processed_dir, f'data_{idx}.pt'))

    def _parse_pdb(self, path):
        atoms = []
        with open(path, 'r') as f:
            for line in f:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    # 提取元素符号
                    element = line[76:78].strip().upper()
                    if not element:
                        atom_name = line[12:16].strip()
                        element = atom_name[0] if atom_name else 'OTHER'
                        if element.isdigit() and len(atom_name) > 1:
                            element = atom_name[1]
                    element = element.split()[0]  # 去除可能的空格
                    
                    # 提取坐标
                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                    except:
                        continue
                        
                    atoms.append((element, (x, y, z)))
        return atoms

    def _create_edges(self, pos):
        num_atoms = pos.size(0)
        edge_index = []
        edge_dist = []
        
        for i in range(num_atoms):
            for j in range(num_atoms):
                if i != j:
                    dist = torch.norm(pos[i] - pos[j])
                    if dist <= self.threshold:
                        edge_index.append([i, j])
                        edge_dist.append(dist)
        print("边数 ",len(edge_index))
        
        if not edge_index:
            return torch.empty((2, 0), dtype=torch.long), torch.empty((0,))
        
        edge_index = torch.tensor(edge_index).t().contiguous()
        edge_dist = torch.tensor(edge_dist, dtype=torch.float)
        return edge_index, edge_dist

    def len(self):
        return len(self.raw_data)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        return data
    
def calculate_metrics(pred_probs, labels):
    """计算分类指标"""
    pred_labels = (pred_probs > 0.5).astype(int)
    
    acc = accuracy_score(labels, pred_labels)
    auc = roc_auc_score(labels, pred_probs)
    f1 = f1_score(labels, pred_labels)
    
    # 计算敏感度和特异度
    tn, fp, fn, tp = confusion_matrix(labels, pred_labels).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    
    return {
        'accuracy': acc,
        'auc': auc,
        'f1': f1,
        'sensitivity': sensitivity,
        'specificity': specificity
    }

from sklearn.model_selection import StratifiedKFold

def main():
    # 数据集参数
    dataset_root = './nrdld_dataset'
    pdb_dir = './pdb_files'
    threshold = 5.0
    
    # 初始化数据集
    dataset = NRDLDDataset(
        root=dataset_root,
        pdb_dir=pdb_dir,
        threshold=threshold
    )
    
    # 交叉验证参数
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    labels = np.array([data.y.item() for data in dataset])
    
    # 模型配置
    node_dim = len(ATOM_TYPES)
    model_configs = {
        'UniMol-GNN': {'class': UniMolGNN, 'args': {'node_dim': node_dim, 'output_dim': 1}},
        'Edge-GNN': {'class': EdgeEnhancedGNN, 'args': {'node_dim': node_dim, 'output_dim': 1}},
        'Basic-GNN': {'class': BasicGNN, 'args': {'node_dim': node_dim, 'output_dim': 1}},
        'MLP': {'class': EnhancedMLP, 'args': {'input_dim': node_dim+3, 'output_dim': 1}}
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 8
    
    # 存储结果
    results = {model_name: {
        'accuracy': [],
        'auc': [],
        'f1': [],
        'sensitivity': [],
        'specificity': []
    } for model_name in model_configs}
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(dataset)), labels)):
        print(f"\n=== Fold {fold_idx+1}/{n_splits} ===")
        
        # 创建子数据集
        train_dataset = dataset[train_idx.tolist()]
        val_dataset = dataset[val_idx.tolist()]
        
        # 计算类别权重
        train_labels = [data.y.item() for data in train_dataset]
        pos_count = sum(train_labels)
        neg_count = len(train_labels) - pos_count
        pos_weight = torch.tensor([neg_count/pos_count], device=device) if pos_count > 0 and neg_count > 0 else torch.tensor([1.0], device=device)
        
        # 数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # 训练所有模型
        for model_name in model_configs:
            print(f"\nTraining {model_name}...")
            
            # 初始化模型
            config = model_configs[model_name]
            model = config['class'](**config['args']).to(device)
            
            # 训练配置
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            
            best_val_auc = 0.0
            patience = 5
            no_improve = 0
            best_weights = None
            
            # 训练循环
            for epoch in range(5):
                model.train()
                train_loss = 0
                all_train_probs = []
                all_train_labels = []
                
                for batch in train_loader:
                    batch = batch.to(device)
                    optimizer.zero_grad()
                    
                    logits = model(batch)
                    y = batch.y.view(-1, 1).float()
                    
                    loss = criterion(logits, y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    
                    probs = torch.sigmoid(logits).cpu().detach().numpy()
                    labels = y.cpu().detach().numpy()
                    
                    train_loss += loss.item() * batch.num_graphs
                    all_train_probs.extend(probs.flatten())
                    all_train_labels.extend(labels.flatten())
                
                # 验证阶段
                model.eval()
                val_loss = 0
                all_val_probs = []
                all_val_labels = []
                
                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(device)
                        logits = model(batch)
                        y = batch.y.view(-1, 1).float()
                        
                        loss = F.binary_cross_entropy_with_logits(logits, y)
                        probs = torch.sigmoid(logits).cpu().numpy()
                        labels = y.cpu().numpy()
                        
                        val_loss += loss.item() * batch.num_graphs
                        all_val_probs.extend(probs.flatten())
                        all_val_labels.extend(labels.flatten())
                
                # 计算指标
                train_metrics = calculate_metrics(np.array(all_train_probs), 
                                                 np.array(all_train_labels))
                val_metrics = calculate_metrics(np.array(all_val_probs),
                                              np.array(all_val_labels))
                
                # 早停机制
                current_auc = val_metrics['auc']
                scheduler.step(val_loss)
                
                if current_auc > best_val_auc:
                    best_val_auc = current_auc
                    no_improve = 0
                    best_weights = model.state_dict().copy()
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        break
                
                print(f"Fold {fold_idx+1} Epoch {epoch+1:02d}: "
                      f"Train Loss: {train_loss/len(train_dataset):.4f} | "
                      f"Val AUC: {current_auc:.4f}")
            
            # 加载最佳模型进行最终验证
            model.load_state_dict(best_weights)
            final_metrics = calculate_metrics(*evaluate(model, val_loader, device))
            
            # 记录结果
            for metric in ['accuracy', 'auc', 'f1', 'sensitivity', 'specificity']:
                results[model_name][metric].append(final_metrics[metric])
    
    # 输出结果
    print("\n交叉验证结果:")
    for model_name in model_configs:
        print(f"\n{model_name}:")
        for metric in ['accuracy', 'auc', 'f1', 'sensitivity', 'specificity']:
            values = results[model_name][metric]
            print(f"  {metric}: {np.mean(values):.4f} ± {np.std(values):.4f}")

def evaluate(model, loader, device):
    model.eval()
    probs = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            y = batch.y.view(-1, 1).float()
            probs.extend(torch.sigmoid(logits).cpu().numpy().flatten())
            labels.extend(y.cpu().numpy().flatten())
    return np.array(probs), np.array(labels)

if __name__ == "__main__":
    main()