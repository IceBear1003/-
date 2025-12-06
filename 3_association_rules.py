import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# 设置中文字体（解决PyCharm绘图中文乱码）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 加载清洗后的数据
df_clean = pd.read_csv("Cleaned_Online_Retail.csv")
df_clean["InvoiceDate"] = pd.to_datetime(df_clean["InvoiceDate"])

# 1. 转换为事务矩阵（一个订单对应一个商品列表）
order_products = df_clean.groupby("InvoiceNo")["StockCode"].apply(list).tolist()
te = TransactionEncoder()
te_matrix = te.fit_transform(order_products)
transaction_df = pd.DataFrame(te_matrix, columns=te.columns_)
print(f"事务矩阵形状：{transaction_df.shape}（{len(transaction_df)}个订单，{transaction_df.shape[1]}个商品）")

# 2. 挖掘频繁项集（作业要求：最小支持度=0.01）
frequent_itemsets = apriori(transaction_df, min_support=0.01, use_colnames=True)
print(f"\n挖掘到频繁项集数量：{len(frequent_itemsets)}")

# 3. 生成强关联规则（按置信度排序，输出前5条）
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)  # 置信度阈值可调整
top5_rules = rules.sort_values("confidence", ascending=False).head(5)

print("\n作业要求：前5条强关联规则（按置信度排序）")
print(top5_rules[["antecedents", "consequents", "support", "confidence", "lift"]].to_string(index=False))

# 4. 频繁项集可视化（热力图，作业要求：热力图/网络图）
# 筛选支持度前10的单个商品频繁项集
top10_single_items = frequent_itemsets[frequent_itemsets["itemsets"].apply(len) == 1].sort_values("support", ascending=False).head(10)
top10_codes = [list(item)[0] for item in top10_single_items["itemsets"]]

# 构建高频商品的事务矩阵子集
top10_transaction = transaction_df[top10_codes]

# 绘制热力图
plt.figure(figsize=(12, 8))
sns.heatmap(top10_transaction.T, cmap="Blues", cbar=False, yticklabels=True)
plt.title("图2-1 高频商品订单共现热力图", fontsize=14)
plt.xlabel("订单序号", fontsize=12)
plt.ylabel("商品ID（StockCode）", fontsize=12)
plt.tight_layout()
plt.savefig("Frequent_Items_Heatmap.png", dpi=300)
plt.close()
print("\n频繁项集热力图已保存为：Frequent_Items_Heatmap.png")