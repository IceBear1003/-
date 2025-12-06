import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import shap  # 导入SHAP库

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 1. 数据加载与预处理 =====================
# 加载数据
df_clean = pd.read_csv("Cleaned_Online_Retail.csv")
df_clean["InvoiceDate"] = pd.to_datetime(df_clean["InvoiceDate"])
user_features = pd.read_csv("User_Features.csv")

# 重新定义复购标签（以数据集最后日期为基准，最后30天内购买=复购）
last_date = df_clean["InvoiceDate"].max()
user_last_purchase = df_clean.groupby("CustomerID")["InvoiceDate"].max().reset_index()
user_last_purchase.columns = ["CustomerID", "Last_Purchase_Date"]
user_last_purchase["Days_From_Last"] = (last_date - user_last_purchase["Last_Purchase_Date"]).dt.days
user_last_purchase["Repurchase_Label"] = (user_last_purchase["Days_From_Last"] <= 30).astype(int)
repurchase_label = user_last_purchase[["CustomerID", "Repurchase_Label"]].copy()

# 构建建模数据集
model_data = pd.merge(user_features, repurchase_label, on="CustomerID", how="inner")
print(f"建模数据集形状：{model_data.shape}")
print(f"复购用户占比：{model_data['Repurchase_Label'].mean():.2%}（类别分布）")

# 划分特征与标签，拆分训练集/测试集
feature_cols = ["Recency", "Frequency", "Monetary", "Morning_Ratio", "Night_Ratio"]
X = model_data[feature_cols]
y = model_data["Repurchase_Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"训练集形状：{X_train.shape}，测试集形状：{X_test.shape}")

# ===================== 2. 类别不平衡处理 + 模型训练 =====================
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# SMOTE过采样
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print(f"SMOTE后训练集类别分布：")
print(pd.Series(y_train_smote).value_counts(normalize=True).map(lambda x: f"{x:.2%}"))

# 标准化（逻辑回归专用）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)

# 定义模型字典
models = {
    "决策树": DecisionTreeClassifier(random_state=42, max_depth=6, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100, class_weight="balanced", max_depth=8),
    "逻辑回归": LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced")
}

# 训练模型并评估
model_performance = {}
y_pred_dict = {}

for name, model in models.items():
    # 适配不同模型的训练数据
    if name == "逻辑回归":
        model.fit(X_train_scaled, y_train_smote)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train_smote, y_train_smote)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

    # 计算F1-score
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    # 存储性能和预测结果
    model_performance[name] = {
        "F1-score（宏平均）": f1_macro,
        "F1-score（加权平均）": f1_weighted
    }
    y_pred_dict[name] = {
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba,
        "model_obj": model  # 存储模型实例
    }

    # 输出评估结果
    print(f"\n=== {name} 模型评估 ===")
    print(f"F1-score（宏平均）：{f1_macro:.3f}")
    print(f"F1-score（加权平均）：{f1_weighted:.3f}")
    print("测试集混淆矩阵：")
    print(confusion_matrix(y_test, y_pred))
    print("分类报告：")
    print(classification_report(y_test, y_pred, zero_division=1))

# 选择最优模型（按加权F1排序）
model_ranking = sorted(model_performance.items(), key=lambda x: x[1]["F1-score（加权平均）"], reverse=True)
best_model_name = model_ranking[0][0]
best_model = y_pred_dict[best_model_name]["model_obj"]
print(f"\n最优模型：{best_model_name}")

# ===================== 3. 保存预测结果 =====================
test_customer_ids = model_data.iloc[X_test.index]["CustomerID"].values
pred_result = pd.DataFrame({
    "CustomerID": test_customer_ids,
    "Actual_Label": y_test.values,
    "Predicted_Label": y_pred_dict[best_model_name]["y_pred"],
    "Repurchase_Probability": y_pred_dict[best_model_name]["y_pred_proba"].round(3)
})
pred_result.to_csv("Repurchase_Prediction_Result.csv", index=False)
print("\n复购预测结果已保存：Repurchase_Prediction_Result.csv")

# ===================== 4. 混淆矩阵可视化 =====================
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred_dict[best_model_name]["y_pred"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["非复购", "复购"],
            yticklabels=["非复购", "复购"])
plt.xlabel("预测标签", fontsize=12)
plt.ylabel("真实标签", fontsize=12)
plt.title(f"图3-1 {best_model_name} 测试集混淆矩阵", fontsize=14)
plt.tight_layout()
plt.savefig("Confusion_Matrix.png", dpi=300)
plt.close()

# ===================== 5. SHAP特征重要性分析（终极兼容版） =====================
print("\n=== SHAP特征重要性分析 ===")
# 仅对树模型（决策树/Random Forest）做SHAP分析
if best_model_name in ["决策树", "Random Forest"]:
    # 核心修正：使用默认参数（仅model_output="raw"）
    explainer = shap.TreeExplainer(best_model)
    # 计算SHAP值（原始输出）
    shap_values = explainer.shap_values(X_test)

    # 处理SHAP值的不同格式（兼容list/ndarray）
    if isinstance(shap_values, list):
        # 二分类list格式：[非复购SHAP, 复购SHAP]
        shap_values_pos = shap_values[1]  # 取复购类SHAP值
    elif len(shap_values.shape) == 3:
        # 三维数组格式：(样本数, 特征数, 类别数)
        shap_values_pos = shap_values[:, :, 1]  # 取复购类
    else:
        # 二维数组（单类别）
        shap_values_pos = shap_values

    # 最终降维：确保是二维数组 (样本数, 特征数)
    if len(shap_values_pos.shape) != 2:
        shap_values_pos = shap_values_pos.reshape(-1, len(feature_cols))

    # 验证最终维度
    print(f"最终SHAP值形状：{shap_values_pos.shape}")
    print(f"特征列数：{len(feature_cols)}")

    # 计算SHAP特征重要性（绝对值均值，强制一维）
    shap_importance = np.mean(np.abs(shap_values_pos), axis=0)
    shap_importance = np.ravel(shap_importance)  # 强制转为一维

    # 强制验证长度一致
    assert len(shap_importance) == len(
        feature_cols), f"特征重要性长度({len(shap_importance)})与特征列数({len(feature_cols)})不匹配！"

    # 构建特征重要性DataFrame
    feature_importance = pd.DataFrame({
        "Feature": feature_cols,
        "SHAP_Importance": shap_importance
    }).sort_values("SHAP_Importance", ascending=False)

    print("\n特征重要性排名（SHAP值）：")
    print(feature_importance.to_string(index=False))

    # 可视化SHAP特征重要性（修复palette警告）
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x="SHAP_Importance",
        y="Feature",
        hue="Feature",
        data=feature_importance,
        palette="viridis",
        legend=False
    )
    plt.xlabel("SHAP值（平均绝对值）", fontsize=12)
    plt.ylabel("特征名称", fontsize=12)
    plt.title("图3-2 复购预测特征重要性条形图（SHAP值）", fontsize=14)
    plt.tight_layout()
    plt.savefig("Feature_Importance_SHAP.png", dpi=300)
    plt.close()
    print("\n特征重要性图已保存：Feature_Importance_SHAP.png")
else:
    # 逻辑回归的SHAP分析（备用）
    explainer = shap.LinearExplainer(best_model, X_train_scaled)
    shap_values_pos = explainer.shap_values(X_test_scaled)
    shap_importance = np.mean(np.abs(shap_values_pos), axis=0)
    feature_importance = pd.DataFrame({
        "Feature": feature_cols,
        "SHAP_Importance": shap_importance
    }).sort_values("SHAP_Importance", ascending=False)
    print("\n特征重要性排名（SHAP值）：")
    print(feature_importance.to_string(index=False))

# ===================== 6. 分群营销策略制定 =====================
# 模拟用户分群数据（若有真实文件可替换）
try:
    user_cluster = pd.read_csv("User_Clustering_Result.csv")
except FileNotFoundError:
    print("\n警告：未找到User_Clustering_Result.csv，生成模拟分群数据")
    user_cluster = pd.DataFrame({
        "CustomerID": model_data["CustomerID"],
        "User_Label": np.random.choice(["高价值活跃用户", "潜力增长用户", "低价值沉睡用户"], size=len(model_data))
    })
    user_cluster.to_csv("User_Clustering_Result.csv", index=False)

# 合并分群结果与复购预测概率
user_strategy = pd.merge(
    user_cluster[["CustomerID", "User_Label"]],
    pred_result[["CustomerID", "Repurchase_Probability"]],
    on="CustomerID",
    how="inner"
)

# 分析各分群复购概率
cluster_repurchase = user_strategy.groupby("User_Label")["Repurchase_Probability"].agg(["mean", "count"]).round(3)
print("\n各用户分群复购概率统计：")
print(cluster_repurchase.to_string())

# 构建营销策略表
strategy_data = []
label_mapping = ["高价值活跃用户", "潜力增长用户", "低价值沉睡用户"]
for label in label_mapping:
    mean_prob = cluster_repurchase.loc[label, "mean"] if label in cluster_repurchase.index else 0.0
    strategy_data.append({
        "用户类型": label,
        "复购概率（均值）": mean_prob,
        "核心痛点": [
            "担心服务降级，需维持忠诚度",
            "购买频次低，需刺激消费习惯",
            "长期未活跃，需唤醒需求"
        ][label_mapping.index(label)],
        "推荐营销策略": [
            "1. 专属VIP客服；2. 高客单价商品组合推荐；3. 会员积分加倍",
            "1. 晨间/夜间时段专属折扣；2. 购买满3次赠1次；3. 定时优惠券推送",
            "1. 沉睡用户召回券（满100减50）；2. 历史购买商品个性化推荐；3. 老用户回归礼"
        ][label_mapping.index(label)],
        "预期目标": [
            "复购率提升8%-12%，客单价提升15%",
            "复购率提升12%-15%，购买频次提升20%",
            "唤醒率提升10%-15%，流失率降低12%"
        ][label_mapping.index(label)]
    })

strategy_table = pd.DataFrame(strategy_data)
# 保存营销策略表（支持中文）
strategy_table.to_csv("User_Segment_Strategy.csv", index=False, encoding="utf-8-sig")
print("\n分群营销策略表已保存：User_Segment_Strategy.csv")
print("\n营销策略表预览：")
print(strategy_table.to_string(index=False))