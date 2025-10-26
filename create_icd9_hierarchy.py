# create_icd9_hierarchy.py
import pandas as pd
from pathlib import Path

# === ĐƯỜNG DẪN ===
PROJECT_ROOT = Path.cwd()
SOURCE_DX = PROJECT_ROOT / "data" / "ccs_multi_dx_tool_2015.csv"
OUTPUT_CSV = PROJECT_ROOT / "data" / "icd9_hierarchy.csv"

# === KIỂM TRA FILE ===
if not SOURCE_DX.exists():
    raise FileNotFoundError(f"Không tìm thấy file: {SOURCE_DX}")

print(f"Đang tạo file icd9_hierarchy.csv từ: {SOURCE_DX}")

# === ĐỌC FILE NGUỒN ===
# AHRQ file dùng tab và có dấu nháy đơn
df = pd.read_csv(SOURCE_DX, sep="\t", dtype=str, engine="python")
df.columns = [c.strip().replace("'", "") for c in df.columns]

# === LỌC VÀ ĐỔI TÊN CỘT ===
cols = [
    'ICD-9-CM CODE',
    'CCS LVL 1', 'CCS LVL 1 LABEL',
    'CCS LVL 2', 'CCS LVL 2 LABEL',
    'CCS LVL 3', 'CCS LVL 3 LABEL',
    'CCS LVL 4', 'CCS LVL 4 LABEL'
]
df = df[[c for c in cols if c in df.columns]]
df.columns = ['ICD9', 'cat1', 'desc1', 'cat2', 'desc2', 'cat3', 'desc3', 'cat4', 'desc4']

# === DỌN DỮ LIỆU ===
for c in df.columns:
    df[c] = df[c].astype(str).str.strip().str.replace("'", "")
df = df[df['ICD9'].notna() & (df['ICD9'].str.strip() != '')]

# === GHI FILE ĐẦY ĐỦ 9 CỘT ===
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Đã lưu: {OUTPUT_CSV}")
print(f"🔹 Tổng số dòng: {len(df)}")
print(f"🔹 Các cột: {list(df.columns)}")
