import pandas as pd
import os

# 查看项目根目录路径（确保数据文件在该路径下）
print("项目根目录：", os.getcwd())

# 读取数据（根据实际文件格式选择，二选一）
# 若为Excel文件（.xlsx）
df = pd.read_excel("Online Retail.xlsx", engine="openpyxl")
# 若为CSV文件（.csv，需指定编码）
# df = pd.read_csv("Online Retail.csv", encoding="latin-1")

# 验证关键字段是否存在（匹配作业要求的7个字段）
required_columns = ["InvoiceNo", "StockCode", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country"]
missing_cols = [col for col in required_columns if col not in df.columns]
if missing_cols:
    print(f"警告：缺失关键字段{missing_cols}，请检查数据集！")
else:
    print("数据加载成功，关键字段完整！")
    print(f"数据形状（行×列）：{df.shape}")
    print("\n前5行数据预览：")
    print(df.head())