import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# ========== 核心修复：设置Matplotlib支持中文 ==========
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认中文字体（黑体）
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
plt.rcParams['font.family'] = 'sans-serif'    # 补充字体族配置
# ====================================================

# 加载清洗后的数据
df_clean = pd.read_csv("Cleaned_Online_Retail.csv")
df_clean["InvoiceDate"] = pd.to_datetime(df_clean["InvoiceDate"], format="%Y-%m-%d %H:%M:%S")

# 1. 转换为事务矩阵（订单-商品）
order_products = df_clean.groupby("InvoiceNo")["StockCode"].apply(list).tolist()
te = TransactionEncoder()
te_matrix = te.fit_transform(order_products)
transaction_df = pd.DataFrame(te_matrix, columns=te.columns_)

# 2. 挖掘频繁项集+强关联规则
frequent_itemsets = apriori(transaction_df, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)
top5_rules = rules.sort_values("confidence", ascending=False).head(5).reset_index(drop=True)

# 3. 提取规则核心信息并转换商品编码为名称（可选）
rule_interpret = top5_rules[["antecedents", "consequents", "support", "confidence", "lift"]].copy()
rule_interpret["antecedents"] = rule_interpret["antecedents"].apply(lambda x: ", ".join(list(x)))
rule_interpret["consequents"] = rule_interpret["consequents"].apply(lambda x: ", ".join(list(x)))
rule_interpret[["support", "confidence", "lift"]] = rule_interpret[["support", "confidence", "lift"]].round(3)

# 补充商品名称映射（若有Description字段）
if "Description" in df_clean.columns:
    product_name = df_clean[["StockCode", "Description"]].drop_duplicates().set_index("StockCode")["Description"].to_dict()
    rule_interpret["antecedents_name"] = rule_interpret["antecedents"].apply(
        lambda x: ", ".join([product_name.get(code, code) for code in x.split(", ")])
    )
    rule_interpret["consequents_name"] = rule_interpret["consequents"].apply(
        lambda x: ", ".join([product_name.get(code, code) for code in x.split(", ")])
    )

# 4. 绘制关联规则网络图（修复后）
G = nx.DiGraph()
# 添加节点和边（取前5条规则）
for idx, row in rule_interpret.iterrows():
    # 优先用商品名称，无名称则用编码
    antecedent = row.get("antecedents_name", row["antecedents"])
    consequent = row.get("consequents_name", row["consequents"])
    G.add_edge(antecedent, consequent, weight=row["confidence"])

# 绘图配置
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, k=3, iterations=20)  # 调整布局参数，避免节点重叠
edges = G.edges()
weights = [G[u][v]["weight"] for u, v in edges]

# 绘制节点、边、标签
nx.draw_networkx_nodes(G, pos, node_size=3000, node_color="#4ECDC4", alpha=0.8)
nx.draw_networkx_edges(G, pos, width=[w*8 for w in weights], alpha=0.6, edge_color="#FF6B6B", arrowstyle="->", arrowsize=10)
nx.draw_networkx_labels(G, pos, font_size=9, font_family="SimHei")  # 明确指定中文字体

# 标题和保存（无中文警告）
plt.title("图2-4 商品关联规则网络图（置信度为权重）", fontsize=14, fontfamily="SimHei")
plt.axis("off")
plt.tight_layout()
plt.savefig("Association_Rules_Network.png", dpi=300, bbox_inches="tight")
plt.close()

print("关联规则网络图已保存（无中文字体警告），可直接插入PDF报告")