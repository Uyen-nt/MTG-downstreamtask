# create_icd9_hierarchy.py
import os
import pickle
from pathlib import Path

PROJECT_ROOT = Path.cwd()
ICD9_TXT = PROJECT_ROOT / "data" / "icd9.txt"
TYPES_FILE = PROJECT_ROOT / "data" / "result" / "mimic3" / "real_mimic3.3digitICD9.types"
OUTPUT_CSV = PROJECT_ROOT / "data" / "icd9_hierarchy.csv"

# Kiểm tra file
if not ICD9_TXT.exists():
    raise FileNotFoundError(f"Không tìm thấy: {ICD9_TXT}")
if not TYPES_FILE.exists():
    raise FileNotFoundError(f"Không tìm thấy: {TYPES_FILE}")

print(f"Đọc từ: {ICD9_TXT}")
print(f"Đọc types từ: {TYPES_FILE}")
print(f"Xuất ra: {OUTPUT_CSV}")

# Đọc icd9.txt
lines = [l.rstrip() for l in ICD9_TXT.read_text(encoding="utf-8").splitlines() if l.strip()]
seen_codes = set()

with open(OUTPUT_CSV, "w", encoding="utf-8") as f:
    f.write("parent,child\n")
    parent = None
    for line in lines:
        if not line.startswith("    "):
            parent = line.split("-")[0].strip()
            seen_codes.add(parent)
        else:
            child = line.strip().split("-")[0].strip()
            f.write(f"{parent},{child}\n")
            seen_codes.add(child)

# ĐỌC .types BẰNG PICKLE
with open(TYPES_FILE, "rb") as f:
    code_to_id = pickle.load(f)

# Thêm mã 3-digit nếu thiếu
added = 0
with open(OUTPUT_CSV, "a", encoding="utf-8") as f:
    for code in code_to_id:
        if "." in code:
            code3 = code.split(".")[0]
            if code3 not in seen_codes:
                f.write(f"{code3},{code3}\n")
                seen_codes.add(code3)
                added += 1

print(f"HOÀN TẤT! Thêm {added} mã 3-digit thiếu.")
print(f"→ {OUTPUT_CSV} có {sum(1 for _ in open(OUTPUT_CSV, encoding='utf-8')) - 1} dòng")
