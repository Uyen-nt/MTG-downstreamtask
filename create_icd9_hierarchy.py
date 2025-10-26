# create_icd9_hierarchy.py
# Tạo icd9_hierarchy.csv từ data/icd9.txt

import os
from pathlib import Path

# ĐƯỜNG DẪN
PROJECT_ROOT = Path(__file__).parent.resolve()
ICD9_TXT = PROJECT_ROOT / "data" / "icd9.txt"
OUTPUT_CSV = PROJECT_ROOT / "data" / "icd9_hierarchy.csv"

# Kiểm tra file tồn tại
if not ICD9_TXT.exists():
    raise FileNotFoundError(f"Không tìm thấy: {ICD9_TXT}")

print(f"Đọc từ: {ICD9_TXT}")
print(f"Xuất ra: {OUTPUT_CSV}")

# Đọc và xử lý
lines = [line.rstrip() for line in ICD9_TXT.read_text().splitlines() if line.strip()]

with open(OUTPUT_CSV, "w", encoding="utf-8") as f:
    f.write("parent,child\n")
    parent = None
    for line in lines:
        if not line.startswith("    "):  # Cấp 1: 001-139
            parent = line.split("-")[0].strip()
        else:  # Cấp 2: 001-009
            child = line.strip().split("-")[0].strip()
            f.write(f"{parent},{child}\n")

print(f"HOÀN TẤT! Tạo {OUTPUT_CSV.name} với {sum(1 for _ in open(OUTPUT_CSV)) - 1} quan hệ cha-con.")
