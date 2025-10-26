# create_icd9_hierarchy.py
import pandas as pd
from pathlib import Path

# === ĐƯỜNG DẪN ===
PROJECT_ROOT = Path.cwd()
SOURCE_DX = PROJECT_ROOT / "data" / "ccs_multi_dx_tool_2015.csv"
OUTPUT_CSV = PROJECT_ROOT / "data" / "icd9_hierarchy.csv"

if not SOURCE_DX.exists():
    raise FileNotFoundError(f"Không tìm thấy file: {SOURCE_DX}")

print(f"Đang tạo file icd9_hierarchy.csv từ: {SOURCE_DX}")

# ==============================
# 1️⃣ ĐỌC FILE GỐC
# ==============================
# Thử các cách đọc khác nhau để tránh lỗi "0 cột"
try:
    df = pd.read_csv(SOURCE_DX, sep="\t", dtype=str, engine="python")
except Exception as e:
    print("❌ Lỗi khi đọc bằng tab, thử lại với dấu phẩy (,)")
    df = pd.read_csv(SOURCE_DX, sep=",", dtype=str, engine="python")

# Nếu vẫn rỗng, in thử vài dòng để kiểm tra
if df.shape[1] == 0:
    print("⚠️ Không đọc được cột nào. In thử 5 dòng đầu:")
    with open(SOURCE_DX, "r", encoding="utf-8", errors="ignore") as f:
        for i in range(5):
            print(f.readline())
    raise ValueError("Không thể đọc được header. Hãy kiểm tra định dạng tệp gốc.")

# ==============================
# 2️⃣ LÀM SẠCH HEADER
# ==============================
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
    raise ValueError(f"Không tìm thấy các cột CCS trong file. Các cột hiện có: {list(df.columns)}")

df = df[cols_in_file]
df.columns = ['ICD9', 'cat1', 'desc1', 'cat2', 'desc2', 'cat3', 'desc3', 'cat4', 'desc4'][:len(cols_in_file)]

# ==============================
# 3️⃣ DỌN DỮ LIỆU
# ==============================
for c in df.columns:
    df[c] = df[c].astype(str).str.strip().str.replace("'", "")

df = df[df['ICD9'].notna() & (df['ICD9'].str.strip() != '')]

# ==============================
# 4️⃣ GHI FILE KẾT QUẢ
# ==============================
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Đã lưu: {OUTPUT_CSV}")
print(f"🔹 Tổng số dòng: {len(df)}")
print(f"🔹 Cột: {list(df.columns)}")
