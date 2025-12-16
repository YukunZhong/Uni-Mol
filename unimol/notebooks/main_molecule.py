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
from models import EdgeEnhancedGNN,BasicGNN,EnhancedMLP,UniMolGNN
torch.serialization.add_safe_globals({"Data": Data})

# 原子类型映射（根据QM9数据集中的元素）
ATOM_TYPES = {
    'H': 0, 
    'C': 1,
    'N': 2,
    'O': 3,
    'F': 4
}

# 修改后的QM9数据加载器
class QM9Dataset(Dataset):
    def __init__(self, root, target_props, threshold=5.0, transform=None, pre_transform=None):
        """
        target_prop: 目标属性选择（homo, lumo, mu, alpha等）
        """
        self.target_props = target_props
        self.threshold = threshold
        self.prop_idx_map = {
            'alpha': 4,   
            'humo': 5,   
            'lumo': 6,    
            'gap': 7,     
            'R2': 8,  
            'zpve': 9     
        }
        self.root = root

        # scaler 持久化文件路径
        self.scaler_path = os.path.join(root, "scaler.pkl")

        # 如果已有 scaler.pkl，就直接加载，否则拟合 + 保存
        if os.path.exists(self.scaler_path):
            self.scaler = joblib.load(self.scaler_path)
        else:
            self.scaler = StandardScaler()
            self._preprocess_targets()
            joblib.dump(self.scaler, self.scaler_path)

        super().__init__(root, transform, pre_transform)
        

    @property
    def raw_file_names(self):
        return [f for f in os.listdir(self.raw_dir) if f.endswith('.xyz')]

    @property
    def processed_file_names(self):
        return [f'data_{i}.pt' for i in range(len(self.raw_paths))]

    def _preprocess_targets(self):
        print("_preprocess_targets start")
        all_targets = []
        for i, path in enumerate(self.raw_paths, 1):
            if i%1000==0 :
                print(f"_preprocess_targets  [{i}/{len(self.raw_paths)}]")
            _, quantum_prop, _, _ = self._parse_file(path)
            vec = [quantum_prop[self.prop_idx_map[p]] for p in self.target_props]
            all_targets.append(vec)
        print("_preprocess_targets done")
        self.scaler.fit(np.array(all_targets))


    def _parse_file(self, path):
        with open(path, 'r') as f:
            lines = f.read().split('\n')
        
        # 第一行是原子数
        num_atoms = int(lines[0].strip())
        
        # 第二行包含属性（格式：gdb_17 C3H8 ... prop1 prop2 ...）
        prop_line = lines[1].split()
        quantum_prop = [float(x) for x in prop_line[2:]]  # 跳过前两个字段
        
        # 原子坐标部分
        atoms_xyz = []
        for line in lines[2:2+num_atoms]:
            if not line.strip():
                continue
            parts = line.split()
            atom_type = parts[0]
            coords = [p.replace('*^', 'e') for p in parts[1:4]]
            x, y, z = map(float, coords)
            atoms_xyz.append((atom_type, (x, y, z)))
        
        # SMILES信息可能在最后几行（具体位置可能变化）
        smiles = lines[-3].split()[0] if len(lines) > num_atoms+4 else ''
        
        return num_atoms, quantum_prop, atoms_xyz, smiles

    def process(self):
        existing = os.listdir(self.processed_dir)
        if len(existing) == len(self.raw_paths):
            return
        
        print("process start")
        
        for i, path in enumerate(self.raw_paths):
            if i%1000==0 :
                print(f"process  [{i}/{len(self.raw_paths)}]")
            num_atoms, quantum_prop, atoms_xyz, _ = self._parse_file(path)
            
            # 生成节点特征
            node_features = []
            pos = []
            for atom_type, coord in atoms_xyz:
                # 原子类型编码（one-hot）
                type_idx = ATOM_TYPES.get(atom_type, 5)  # 未知原子类型设为5
                one_hot = F.one_hot(torch.tensor(type_idx), num_classes=6).float()
                node_features.append(one_hot)
                pos.append(coord)
            
            x = torch.stack(node_features)
            pos = torch.tensor(pos, dtype=torch.float)
            
            # 生成边索引
            edge_index, edge_dist = self._create_edges(pos)
            
            # 目标属性
            target_vec = [ quantum_prop[self.prop_idx_map[p]] for p in self.target_props ]
            y = torch.from_numpy(
                    self.scaler.transform(np.array(target_vec).reshape(1,-1))
                ).float().squeeze(0)    # shape = (6,)
            data = Data(x=x, pos=pos, edge_index=edge_index,
                        edge_dist=edge_dist, y=y)
            
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
                
            if self.pre_transform is not None:
                data = self.pre_transform(data)
                
            torch.save(data, os.path.join(self.processed_dir, f'data_{i}.pt'))

        print("process done")

    def _create_edges(self, pos):
        # 基于距离阈值构建边
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
        
        if len(edge_index) == 0:
            return torch.empty((2, 0), dtype=torch.long), torch.empty((0,))
        
        edge_index = torch.tensor(edge_index).t().contiguous()
        edge_dist = torch.tensor(edge_dist, dtype=torch.float)
        return edge_index, edge_dist

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        return data

# 修改后的模型训练流程
def main():
    # 数据集参数
    dataset_root = './qm9_dataset'
    target_propertys = ['homo','lumo','gap','mu','alpha','cv']  # 可更改为其他属性
    threshold = 5.0
    
    # 初始化数据集
    dataset = QM9Dataset(
        root=dataset_root,
        target_props=target_propertys,
        threshold=threshold
    )
    
    # 划分训练集、验证集、测试集
    train_idx, test_idx = train_test_split(
        range(len(dataset)),
        test_size=0.2,
        random_state=42
    )
    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=0.125,  # 0.8*0.125=0.1
        random_state=42
    )
    
    train_dataset = dataset[train_idx]
    val_dataset = dataset[val_idx]
    test_dataset = dataset[test_idx]
    
    # 创建数据加载器
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # print(f"总样本数: {len(dataset)}")
    # data0 = dataset[0]
    # print(data0)      # Data(x=[N,6], edge_index=[2,E], pos=[N,3], edge_dist=[E], y=[6]) 

    
    # 初始化模型（使用之前定义的UniMolGNN、BasicGNN、EnhancedMLP）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    models = {
        'UniMol-GNN': UniMolGNN(node_dim=6).to(device),
        'Edge-GNN': EdgeEnhancedGNN(node_dim=6).to(device),
        'Basic-GNN': BasicGNN(node_dim=6).to(device),
        'MLP': EnhancedMLP(input_dim=6+3).to(device)
    }
    
    # 训练配置
    epochs = 100
    results = {}
    
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-4,           
            weight_decay=1e-5 
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        patience = 10
        no_improve = 0
        
        for epoch in range(epochs):
            # 训练阶段
            model.train()
            epoch_train_loss = 0
            for i, batch in enumerate(train_loader, 1):
                batch = batch.to(device)
                optimizer.zero_grad()
                pred = model(batch)
                y = batch.y.view(batch.num_graphs, -1)
                loss = F.mse_loss(pred, y)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item() * batch.num_graphs

                if i % 100 == 0:
                    print(f"{model_name} [Epoch {epoch+1} Batch {i}/{len(train_loader)}]  batch_loss = {loss.item():.4f}")
 
            train_loss = epoch_train_loss / len(train_dataset)
            
            # 验证阶段
            model.eval()
            epoch_val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    pred = model(batch)
                    y = batch.y.view(batch.num_graphs, -1)
                    loss = F.mse_loss(pred, y)
                    epoch_val_loss += loss.item() * batch.num_graphs
            val_loss = epoch_val_loss / len(val_dataset)
            
            # 记录和学习率调整
            scheduler.step(val_loss)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve = 0
                torch.save(model.state_dict(), f'best_{model_name}.pth')
                print(f"Epoch {epoch+1}: val_loss improved to {val_loss:.4f}, saving model.")
            else:
                no_improve += 1
                print(f"Epoch {epoch+1}: no improvement ({no_improve}/{patience}).")
                if no_improve >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    break
            
            print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # 测试最佳模型
        model.load_state_dict(torch.load(f'best_{model_name}.pth'))
        test_loss = evaluate(model, test_loader, device)
        results[model_name] = {
            'train_loss': train_losses,
            'val_loss': val_losses,
            'test_loss': test_loss
        }
        print(f"\n{model_name} Final Test Loss: {test_loss:.4f}")
    
    # 结果对比
    print("\nPerformance Comparison:")
    for model_name, metrics in results.items():
        print(f"{model_name}:")
        print(f"  Best Val Loss: {min(metrics['val_loss']):.4f}")
        print(f"  Final Test Loss: {metrics['test_loss']:.4f}")

def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)
            y = batch.y.view(batch.num_graphs, -1)
            loss = F.mse_loss(pred, y)
            total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)

if __name__ == "__main__":
    main()