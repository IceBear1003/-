import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 加载用户特征表
user_features = pd.read_csv("User_Features.csv")

# 1. 数据预处理（RFM特征标准化，排除异常值）
rfm_data = user_features[["Recency", "Frequency", "Monetary"]].copy()

# 用IQR法则处理极端值（避免聚类偏差）
for col in rfm_data.columns:
    Q1 = rfm_data[col].quantile(0.25)
    Q3 = rfm_data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    rfm_data = rfm_data[(rfm_data[col] >= lower) & (rfm_data[col] <= upper)]

# 特征标准化
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_data)

# 2. 确定最优K值（轮廓系数法，作业要求至少3类）
silhouette_scores = []
k_range = range(3, 6)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)  # n_init=10避免警告
    labels = kmeans.fit_predict(rfm_scaled)
    score = silhouette_score(rfm_scaled, labels)
    silhouette_scores.append(score)
    print(f"K={k}时轮廓系数：{score:.3f}")

# 选择轮廓系数最高的K值（假设K=3最优）
best_k = k_range[silhouette_scores.index(max(silhouette_scores))]
print(f"\n最优聚类数量K={best_k}")

# 3. 执行K-Means聚类并定义用户标签
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
user_features["Cluster_Label"] = kmeans.fit_predict(scaler.transform(user_features[["Recency", "Frequency", "Monetary"]]))

# 分析各聚类的RFM特征，定义标签（作业要求至少3类）
cluster_analysis = user_features.groupby("Cluster_Label")[["Recency", "Frequency", "Monetary"]].mean()
print("\n各聚类RFM特征均值（用于标签定义）：")
print(cluster_analysis)

# 定义用户标签（基于业务逻辑）
label_mapping = {
    0: "高价值活跃用户",  # Recency小、Frequency高、Monetary高
    1: "潜力增长用户",    # Recency中等、Frequency中等、Monetary中等
    2: "低价值沉睡用户"  # Recency大、Frequency低、Monetary低
}
user_features["User_Label"] = user_features["Cluster_Label"].map(label_mapping)

# 统计各标签占比
label_count = user_features["User_Label"].value_counts(normalize=True).round(3) * 100
print("\n各用户标签占比：")
for label, ratio in label_count.items():
    print(f"{label}：{ratio:.1f}%")

# 保存分群结果
user_features.to_csv("User_Clustering_Result.csv", index=False)
print("\n用户分群结果已保存为：User_Clustering_Result.csv")

# 4. 聚类结果可视化（作业要求：三维散点图/雷达图）
# 4.1 三维散点图（RFM三维空间分布）
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]  # 三类用户的颜色
labels = user_features["User_Label"].unique()

for i, label in enumerate(labels):
    label_data = user_features[user_features["User_Label"] == label]
    ax.scatter(
        label_data["Recency"],
        label_data["Frequency"],
        label_data["Monetary"],
        c=colors[i],
        label=label,
        alpha=0.6,
        s=50
    )

ax.set_xlabel("Recency（最近购买天数）", fontsize=12)
ax.set_ylabel("Frequency（购买次数）", fontsize=12)
ax.set_zlabel("Monetary（总消费金额）", fontsize=12)
ax.set_title("图2-2 RFM用户分群三维散点图", fontsize=14)
ax.legend()
plt.savefig("RFM_3D_Cluster.png", dpi=300, bbox_inches="tight")
plt.close()

# 4.2 雷达图（标准化特征对比）
# 计算各聚类标准化后的RFM均值
cluster_scaled_mean = pd.DataFrame(
    scaler.transform(cluster_analysis),
    columns=["Recency", "Frequency", "Monetary"],
    index=cluster_analysis.index
)
# Recency为“越小越好”，取负转换为“越大越好”（与其他指标方向一致）
cluster_scaled_mean["Recency"] = -cluster_scaled_mean["Recency"]

# 绘制雷达图
categories = ["Recency（负向转换）", "Frequency", "Monetary"]
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]  # 闭合雷达图

fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
for i, (cluster, color) in enumerate(zip(cluster_scaled_mean.index, colors)):
    values = cluster_scaled_mean.loc[cluster].values.tolist()
    values += values[:1]  # 闭合数据
    ax.plot(angles, values, color=color, linewidth=2, label=label_mapping[cluster])
    ax.fill(angles, values, color=color, alpha=0.25)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=12)
ax.set_title("图2-3 RFM用户分群雷达图（标准化特征）", fontsize=14, pad=20)
ax.legend(loc="upper right")
plt.savefig("RFM_Radar_Cluster.png", dpi=300, bbox_inches="tight")
plt.close()
print("用户分群可视化图表已保存：RFM_3D_Cluster.png、RFM_Radar_Cluster.png")