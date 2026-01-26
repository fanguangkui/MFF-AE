import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import adjusted_rand_score, silhouette_score
from DetectOutlier.utils.data import get_distance, get_scale, get_global
# 设置随机种子以确保可重复性
np.random.seed(42)

# 生成示例数据
def load_data(data_name, remove_list = ()):
    dir_root = f"../datasets/Format"
    data = pd.read_csv(f"{dir_root}/{data_name}.csv", index_col=0)
    Protein_name = data.columns.values
    Sample_name = data.index.values
    Proteomic_data = get_global(data, strategy="global_min")

    return Proteomic_data.values, Protein_name, Sample_name
# 层次聚类
dataset = ('HCC_P', 'HCC_T')
P_data, P_protein_name, P_sample_name = load_data(dataset[0])
T_data, T_protein_name, T_sample_name = load_data(dataset[1])
X = np.concatenate([P_data, T_data])
y_true = np.concatenate([np.array([0 for _ in range(P_data.shape[0])]), np.array([1 for _ in range(T_data.shape[0])])])
Z = linkage(X, method='ward')

# 绘制树状图
plt.figure(figsize=(12, 6))
plt.title('层次聚类树状图')
plt.xlabel('样本索引')
plt.ylabel('距离')
dendrogram(Z, truncate_mode='lastp', p=12, show_leaf_counts=True, show_contracted=True)
plt.show()

# 根据树状图切割成2个聚类
y_pred = fcluster(Z, t=2, criterion='maxclust')

# 评估聚类质量（与真实标签比较）
ari = adjusted_rand_score(y_true, y_pred)
silhouette_avg = silhouette_score(X, y_pred)

print(f"调整兰德指数 (ARI): {ari:.3f}")
print(f"轮廓系数: {silhouette_avg:.3f}")

