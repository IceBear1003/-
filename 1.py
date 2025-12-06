import pandas as pd
import os

# 解决中文路径/文件名可能的编码问题
os.environ['PYTHONIOENCODING'] = 'utf-8'

# 1. 加载原始数据与预处理后数据
# 建议将文件放在代码同目录，或填写完整路径（示例：r"C:\Users\胡金孔\Desktop\Online Retail.xlsx"）
try:
    df_raw = pd.read_excel("Online Retail.xlsx", engine="openpyxl")  # 原始数据
    df_clean = pd.read_csv("Cleaned_Online_Retail.csv")  # 预处理后数据
except FileNotFoundError as e:
    print(f"文件未找到！请检查路径：{e}")
    print("提示：请将 Online Retail.xlsx 和 Cleaned_Online_Retail.csv 放在代码同目录下")
    exit()

# 2. 计算行数与留存率
raw_rows = df_raw.shape[0]  # 原始数据总行数
clean_rows = df_clean.shape[0]  # 预处理后数据总行数
deleted_rows = raw_rows - clean_rows  # 删除的无效样本数
retention_rate = (clean_rows / raw_rows) * 100  # 数据留存率（百分比）

# 3. 输出结果（用于报告引用）
print("=== 预处理前后数据行数对比 ===")
print(f"原始数据集行数：{raw_rows:,}")  # 千位分隔符，更易读
print(f"预处理后数据集行数：{clean_rows:,}")
print(f"删除无效样本数：{deleted_rows:,}")
print(f"数据留存率：{retention_rate:.1f}%")

# 生成行数对比表（用于PDF报告插入）
row_comparison = pd.DataFrame({
    "数据阶段": ["原始数据集", "预处理后数据集"],
    "行数": [raw_rows, clean_rows],
    "无效样本数": ["-", deleted_rows],
    "留存率": ["100.0%", f"{retention_rate:.1f}%"]
})

# Pandas样式格式化（符合作业“禁用截图表格”要求）
styled_row_table = row_comparison.style.set_caption("表1-1 预处理前后数据行数对比") \
    .set_table_styles([{"selector": "caption", "props": [("font-size", "12px"), ("font-weight", "bold")]}]) \
    .set_properties(**{"text-align": "center", "font-size": "11px"})

# 保存为HTML（可直接复制到PDF报告，或在Notebook中展示）
# 解决HTML保存的编码问题（中文不乱码）
styled_row_table.to_html("row_comparison.html", index=False, encoding='utf-8')
print("\n行数对比表已保存为HTML文件（UTF-8编码），可直接复制到PDF报告")

# 额外：输出纯文本表格（备用，无样式依赖）
print("\n=== 纯文本版对比表（无依赖）===")
print(row_comparison.to_string(index=False))