# create_icd9_hierarchy.py
import pickle
from pathlib import Path

# ĐƯỜNG DẪN
PROJECT_ROOT = Path.cwd()
HIERARCHY_2COL = PROJECT_ROOT / "data" / "icd9_hierarchy.csv"
TYPES_FILE = PROJECT_ROOT / "data" / "result" / "mimic3" / "real_mimic3.3digitICD9.types"
OUTPUT_9COL = PROJECT_ROOT / "data" / "icd9_hierarchy_full.csv"

# KIỂM TRA FILE
for f in [HIERARCHY_2COL, TYPES_FILE]:
    if not f.exists():
        raise FileNotFoundError(f"Không tìm thấy: {f}")

print(f"Đang tạo file 9 cột từ:")
print(f"  - {HIERARCHY_2COL}")
print(f"  - {TYPES_FILE}")
print(f"  → {OUTPUT_9COL}")

# 1. ĐỌC HIERARCHY 2 CỘT → XÂY BẢNG CHA
parent_of = {}  # child → parent
with open(HIERARCHY_2COL, "r", encoding="utf-8") as f:
    next(f)  # skip header
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 2:
            continue
        parent, child = parts[0].strip(), parts[1].strip()
        parent_of[child] = parent

# 2. ĐỌC TYPES → LẤY TẤT CẢ MÃ ICD-9
with open(TYPES_FILE, "rb") as f:
    code_to_id = pickle.load(f)

# 3. TẠO FILE 9 CỘT
with open(OUTPUT_9COL, "w", encoding="utf-8") as f:
    # Header
    f.write("ICD9,cat1,desc1,cat2,desc2,cat3,desc3,cat4,desc4\n")
    
    count = 0
    for code_str in code_to_id.keys():
        if "." not in code_str:
            continue  # bỏ mã không hợp lệ
        
        code3 = code_str.split(".")[0]
        parent = parent_of.get(code3, "")
        
        # Tạo dòng 9 cột
        icd9 = f'"{code_str}"'
        cat1 = f'"{parent}"' if parent else '""'
        desc1 = f'"A_{parent}"' if parent else '""'
        # Cấp cao hơn: để trống
        cat2 = cat3 = cat4 = '""'
        desc2 = desc3 = desc4 = '""'
        
        line = f"{icd9},{cat1},{desc1},{cat2},{desc2},{cat3},{desc3},{cat4},{desc4}\n"
        f.write(line)
        count += 1

print(f"HOÀN TẤT! Tạo {OUTPUT_9COL}")
print(f"   → {count} dòng mã ICD-9")
print(f"   → Sẵn sàng cho build_trees.py gốc")
