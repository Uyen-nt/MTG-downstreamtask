# create_icd9_hierarchy.py
import pandas as pd
import re
from pathlib import Path

# === ĐƯỜNG DẪN ===
PROJECT_ROOT = Path.cwd()
SOURCE_DX = PROJECT_ROOT / "data" / "ccs_multi_dx_tool_2015.csv"
OUTPUT_CSV = PROJECT_ROOT / "data" / "icd9_hierarchy.csv"

if not SOURCE_DX.exists():
    raise FileNotFoundError(f"Không tìm thấy file: {SOURCE_DX}")

print(f"Đang tạo file icd9_hierarchy.csv từ: {SOURCE_DX}")

# === ĐỌC DÒNG ĐẦU ĐỂ PHÂN TÍCH ĐỊNH DẠNG ===
with open(SOURCE_DX, "r", encoding="utf-8", errors="ignore") as f:
    header = f.readline().strip()
    preview = [header] + [f.readline().strip() for _ in range(2)]

print("📄 Dòng đầu tiên trong file:")
print(header)

# === XÁC ĐỊNH DELIMITER ===
if "\t" in header:
    sep = "\t"
elif header.count(",") >= 3:
    sep = ","
else:
    # fallback: split bằng regex theo nhiều dấu cách hoặc tab
    sep = None  # sẽ xử lý bằng regex sau

print(f"🔍 Dự đoán delimiter: {repr(sep)}")

# === TRƯỜNG HỢP 1: CSV hoặc TSV BÌNH THƯỜNG ===
if sep is not None:
    df = pd.read_csv(SOURCE_DX, sep=sep, dtype=str, engine="python")
# === TRƯỜNG HỢP 2: FILE CÓ DẤU NHÁY ĐƠN VÀ KHÔNG DELIMITER RÕ RÀNG ===
else:
    rows = []
    with open(SOURCE_DX, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Tách theo pattern: giữa các cặp dấu nháy đơn
            parts = re.findall(r"'(.*?)'", line)
            if parts:
                rows.append(parts)
    df = pd.DataFrame(rows[1:], columns=rows[0])

# === LÀM SẠCH TÊN CỘT ===
df.columns = [c.strip().replace("'", "") for c in df.columns]

needed_cols = [
    'ICD-9-CM CODE',
    'CCS LVL 1', 'CCS LVL 1 LABEL',
    'CCS LVL 2', 'CCS LVL 2 LABEL',
    'CCS LVL 3', 'CCS LVL 3 LABEL',
    'CCS LVL 4', 'CCS LVL 4 LABEL'
]
cols_in_file = [c for c in needed_cols if c in df.columns]
if len(cols_in_file) == 0:
    raise ValueError(f"❌ Không tìm thấy các cột CCS. Các cột hiện có: {list(df.columns)}")

df = df[cols_in_file]
df.columns = ['ICD9', 'cat1', 'desc1', 'cat2', 'desc2', 'cat3', 'desc3', 'cat4', 'desc4'][:len(cols_in_file)]

# === DỌN DỮ LIỆU ===
for c in df.columns:
    df[c] = df[c].astype(str).str.strip().str.replace("'", "")

df = df[df['ICD9'].notna() & (df['ICD9'].str.strip() != '')]

# === GHI FILE ===
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Đã lưu: {OUTPUT_CSV}")
print(f"🔹 Tổng số dòng: {len(df)}")
print(f"🔹 Cột: {list(df.columns)}")
