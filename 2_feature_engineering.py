import pandas as pd
import numpy as np

# 加载清洗后的数据
df_clean = pd.read_csv("Cleaned_Online_Retail.csv")
# 1. 时间格式预处理（修正为实际的ISO8601格式：年-月-日 时:分:秒）
# 关键修改：format参数改为"%Y-%m-%d %H:%M:%S"，匹配数据中的"2010-12-01 08:26:00"格式
df_clean["InvoiceDate"] = pd.to_datetime(df_clean["InvoiceDate"], format="%Y-%m-%d %H:%M:%S")
analysis_end_date = df_clean["InvoiceDate"].max()  # 分析截止日期（最后一笔订单日期）
print(f"分析截止日期：{analysis_end_date}")
# 后续RFM计算、时段偏好构建代码保持不变（无需修改）
# 2.1 RFM指标计算
user_rfm = df_clean.groupby("CustomerID").agg(
    Recency=("InvoiceDate", lambda x: (analysis_end_date - x.max()).days),  # 最近购买天数
    Frequency=("InvoiceNo", "nunique"),  # 购买次数（去重订单号）
    Monetary=("Quantity", lambda x: (x * df_clean.loc[x.index, "UnitPrice"]).sum())  # 总消费金额
).reset_index()
# 2.2 购买时段偏好（晨间6:00-12:00，夜间18:00-24:00）
df_clean["Hour"] = df_clean["InvoiceDate"].dt.hour
df_clean["Time_Period"] = np.where(
    df_clean["Hour"].between(6, 11), "Morning",
    np.where(df_clean["Hour"].between(18, 23), "Night", "Other")
)
# 计算用户各时段订单占比
user_time_preference = df_clean.groupby("CustomerID")["Time_Period"].value_counts(normalize=True).unstack(fill_value=0)
user_time_preference = user_time_preference[["Morning", "Night"]].rename(
    columns={"Morning": "Morning_Ratio", "Night": "Night_Ratio"}
).reset_index()
# 合并RFM与时段偏好，形成完整用户特征表
user_features = pd.merge(user_rfm, user_time_preference, on="CustomerID")
print(f"\n用户特征表形状：{user_features.shape}（{len(user_features)}个用户，{user_features.shape[1]}个特征）")
user_features.to_csv("User_Features.csv", index=False)
print("用户特征表已保存为：User_Features.csv")

# 3. 构建商品级特征（作业要求：商品被购买频次、平均订单量）
product_features = df_clean.groupby("StockCode").agg(
    Purchase_Frequency=("CustomerID", "nunique"),  # 商品被购买的用户数（频次）
    Avg_Order_Quantity=("Quantity", "mean")  # 商品平均每订单购买量
).reset_index()
# 过滤低频商品（避免噪声，可选）
product_features = product_features[product_features["Purchase_Frequency"] >= 3]
print(f"\n商品特征表形状：{product_features.shape}（{len(product_features)}个有效商品）")
product_features.to_csv("Product_Features.csv", index=False)
print("商品特征表已保存为：Product_Features.csv")