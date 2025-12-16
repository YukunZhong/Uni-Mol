import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

# 稀疏方法和中断概率标签
sparse_methods = [
    'best_val_loss', 'final_test_loss'
]
interrupt_labels = ['UniMol-GNN', 'Edge-GNN', 'Basic-GNN', 'MLP']
colors = ['#427ab2', '#f09148', '#ff9896', '#afc7e8']

# 表格数据 (每行是一个 sparse method，对应不同中断概率的值)
# data = [
#     [0.73, 0.41, 0.27, 0.20, 0.18, 0.10],  # original_pymarl2
#     [0.53, 0.45, 0.41, 0.35, 0.27, 0.27],  # relu_sae
#     [0.90, 0.69, 0.39, 0.37, 0.33, 0.27],  # naive_sae
#     [0.47, 0.37, 0.36, 0.27, 0.37, 0.30],  # jumprelu_sae
#     [0.55, 0.49, 0.29, 0.18, 0.16, 0.16],  # topk_sae
#     [0.88, 0.82, 0.67, 0.39, 0.29, 0.31],  # improved_topk_sae
# ]

# data = [
#     [0.88, 0.71, 0.24, 0.16, 0.18, 0.02],  # original_pymarl2
#     [0.14, 0.10, 0.18, 0.10, 0.10, 0.10],  # relu_sae
#     [0.90, 0.84, 0.88, 0.39, 0.24, 0.04],  # naive_sae
#     [0.12, 0.12, 0.14, 0.16, 0.16, 0.16],  # jumprelu_sae
#     [0.94, 0.90, 0.80, 0.55, 0.10, 0.00],  # topk_sae
#     [0.88, 0.86, 0.55, 0.29, 0.18, 0.08],  # improved_topk_sae
# ]

data = [
    [0.0329, 0.2441, 0.3906, 0.4307],  # original_pymarl2
    [0.0334, 0.2451, 0.3925, 0.4350]  # relu_sae
    # [0.86, 0.90, 0.84, 0.73, 0.65, 0.51],  # naive_sae
    # [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],  # jumprelu_sae
    # [0.82, 0.76, 0.65, 0.65, 0.47, 0.33],  # topk_sae
    # [0.84, 0.76, 0.73, 0.82, 0.67, 0.65],  # improved_topk_sae
]

data = np.array(data)
x = np.arange(len(sparse_methods))
bar_width = 0.20

plt.figure(figsize=(14, 8))

# 为每个中断概率绘图
for i in range(len(interrupt_labels)):
    offset = (i - 2.5) * bar_width
    plt.bar(x + offset, data[:, i], width=bar_width, color=colors[i], label=interrupt_labels[i])
    for xi, val in zip(x + offset, data[:, i]):
        plt.text(xi, val + 0.01, f'{val:.4f}', ha='center', va='bottom', fontsize=8)

# 图形设置
# plt.xlabel('Sparse Methods', fontsize=12)
plt.ylabel('MSE', fontsize=12)
plt.title('Validation MSE and Test MSE for different models', fontsize=14)
plt.xticks(x, sparse_methods, rotation=15)
plt.ylim(0, 0.5)
plt.grid(True, linestyle='--', alpha=0.6)

# 图例
legend_patches = [mpatches.Patch(color=colors[i], label=f'{interrupt_labels[i]}') for i in range(len(interrupt_labels))]
plt.legend(handles=legend_patches, title='Different Models', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig('mse_result.png', bbox_inches='tight')
plt.close()
