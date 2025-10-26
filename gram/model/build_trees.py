# gram/model/build_trees.py (TỐI ƯU)
import sys
import pickle
from collections import defaultdict

def build_tree(hierarchy_file, seqs_file, types_file, output_prefix):
    print(f"Loading hierarchy from {hierarchy_file}...")
    parent_to_children = defaultdict(list)
    code_set = set()

    with open(hierarchy_file, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split(',')
            if len(parts) < 2: continue
            parent, child = parts[0].strip(), parts[1].strip()
            parent_to_children[parent].append(child)
            code_set.add(parent)
            code_set.add(child)

    # TỐI ƯU: XÂY BẢNG CHA
    parent_of = {}
    for parent, children in parent_to_children.items():
        for child in children:
            parent_of[child] = parent

    print(f"Loading types from {types_file}...")
    with open(types_file, 'rb') as f:
        code_to_id = pickle.load(f)
    id_to_code3 = {v: k.split('.')[0] for k, v in code_to_id.items() if '.' in k}

    print(f"Loading sequences from {seqs_file}...")
    with open(seqs_file, 'rb') as f:
        seqs = pickle.load(f)

    print("Building tree from real data...")
    tree = defaultdict(set)  # DÙNG SET ĐỂ TỰ LOẠI TRÙNG
    total_codes = 0

    for seq in seqs:
        for visit in seq:
            for code_id in visit:
                if code_id not in id_to_code3: continue
                code3 = id_to_code3[code_id]
                if code3 not in code_set: continue

                # TRA CỨU NHANH
                current = code3
                ancestors = []
                while current:
                    ancestors.append(current)
                    if current not in parent_of:
                        break
                    current = parent_of[current]

                # Thêm cạnh
                for i in range(len(ancestors) - 1):
                    tree[ancestors[i+1]].add(ancestors[i])
                total_codes += 1

    # Chuyển set → list
    for k in tree:
        tree[k] = list(tree[k])

    # Phân cấp
    levels = {}
    for code in code_set:
        depth = 0
        current = code
        while current in parent_of:
            depth += 1
            current = parent_of[current]
        levels.setdefault(depth, []).append(code)

    # Lưu
    for level, codes in levels.items():
        with open(f"{output_prefix}.level{level}.pk", 'wb') as f:
            pickle.dump(codes, f)
        print(f"Saved level {level}: {len(codes)} codes")

    print(f"Tree saved to {output_prefix}.level*.pk")
    print(f"Processed {total_codes} code occurrences")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python build_trees.py <hierarchy.csv> <seqs> <types> <output_prefix>")
        sys.exit(1)
    build_tree(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
