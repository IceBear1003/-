# 补充前置：导入库+数据加载+变量定义
import numpy as np
import pandas as pd
import os

# 1. 数据加载（确保数据集在项目根目录）
print("项目根目录：", os.getcwd())
# 读取数据（根据实际文件格式选择，二选一）
# 若为Excel文件
df = pd.read_excel("Online Retail.xlsx", engine="openpyxl")
# 若为CSV文件
# df = pd.read_csv("Online Retail.csv", encoding="latin-1")

# 2. 定义关键字段列表（required_columns）
required_columns = ["InvoiceNo", "StockCode", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country"]

# 1. 缺失值处理（作业要求：删除CustomerID缺失记录）
print(f"\n清洗前各字段缺失值统计：")
print(df[required_columns].isnull().sum())

df_clean = df.dropna(subset=["CustomerID"]).copy()
print(f"删除CustomerID缺失记录后，数据量：{len(df_clean)}")

# 2. 异常值处理（作业要求：删除负数量、0单价订单）
# 查看异常值分布
print(f"\n清洗前Quantity取值范围：{df_clean['Quantity'].min()} ~ {df_clean['Quantity'].max()}")
print(f"清洗前UnitPrice取值范围：{df_clean['UnitPrice'].min()} ~ {df_clean['UnitPrice'].max()}")

# 筛选有效订单（数量>0，单价>0）
df_clean = df_clean[(df_clean["Quantity"] > 0) & (df_clean["UnitPrice"] > 0)]
print(f"删除异常值后，最终清洗数据量：{len(df_clean)}")
print(f"总数据清洗率：{(len(df) - len(df_clean))/len(df):.2%}")

# 3. 保存清洗后数据（作业要求输出.csv文件）
df_clean.to_csv("Cleaned_Online_Retail.csv", index=False, encoding="utf-8-sig")
print("\n清洗后数据已保存为：Cleaned_Online_Retail.csv")