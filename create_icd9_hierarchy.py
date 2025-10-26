# tạo icd9_hierarchy.csv
import os
from pathlib import Path

PROJECT_ROOT = Path.cwd()
ICD9_TXT = PROJECT_ROOT / "data" / "icd9.txt"
TYPES_FILE = PROJECT_ROOT / "data" / "result" / "mimic3" / "real_mimic3.3digitICD9.types"
OUTPUT_CSV = PROJECT_ROOT / "data" / "icd9_hierarchy.csv"

# Đọc icd9.txt
lines = [l.rstrip() for l in ICD9_TXT.read_text().splitlines() if l.strip()]
seen_codes = set()

with open(OUTPUT_CSV, "w") as f:
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

# Thêm các mã 3-digit từ types nếu thiếu
with open(TYPES_FILE) as f:
    for line in f:
        code = line.strip().split("\t")[0].split(".")[0]
        if code not in seen_codes:
            with open(OUTPUT_CSV, "a") as f2:
                f2.write(f"{code},{code}\n")

print(f"HOÀN TẤT: {OUTPUT_CSV} có {sum(1 for _ in open(OUTPUT_CSV))-1} dòng")
