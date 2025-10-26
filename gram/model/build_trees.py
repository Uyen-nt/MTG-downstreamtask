# gram/model/build_trees.py
# TƯƠNG THÍCH VỚI icd9_hierarchy.csv (2 cột: parent,child)
import sys
import pickle
from collections import defaultdict

def build_tree(hierarchy_file, seqs_file, types_file, output_prefix):
    print(f"Loading hierarchy from {hierarchy_file}...")
    parent_to_children = defaultdict(list)
    code_set = set()

    with open(hierarchy_file, 'r', encoding='utf-8') as f:
        next(f)  # skip header: parent,child
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 2:
                continue
            parent, child = parts[0].strip(), parts[1].strip()
            parent_to_children[parent].append(child)
            code_set.add(parent)
            code_set.add(child)

    print(f"Loaded {len(code_set)} unique codes from hierarchy.")

    print(f"Loading types from {types_file}...")
    with open(types_file, 'rb') as f:
        code_to_id = pickle.load(f)
    id_to_code3 = {v: k.split('.')[0] for k, v in code_to_id.items() if '.' in k}

    print(f"Loading sequences from {seqs_file}...")
    with open(seqs_file, 'rb') as f:
        seqs = pickle.load(f)

    print("Building tree structure from real data...")
    tree = defaultdict(list)
    for seq in seqs:
        for visit in seq:
            for code_id in visit:
                if code_id not in id_to_code3:
                    continue
                code3 = id_to_code3[code_id]
                if code3 not in code_set:
                    continue
                # Tìm đường từ leaf → root
                path = []
                current = code3
                while current:
                    path.append(current)
                    if current not in parent_to_children:
                        break
                    parents = parent_to_children[current]
                    if not parents:
                        break
                    current = parents[0]  # giả sử 1 cha
                # Thêm cạnh
                for i in range(len(path) - 1):
                    tree[path[i+1]].append(path[i])

    # Loại trùng
    for parent in tree:
        tree[parent] = list(set(tree[parent]))

    # Phân cấp theo độ sâu
    levels = {}
    for code in code_set:
        depth = 0
        current = code
        while current in parent_to_children and parent_to_children[current]:
            depth += 1
            current = parent_to_children[current][0]
        levels.setdefault(depth, []).append(code)

    # Lưu theo level
    for level, codes in levels.items():
        output_file = f"{output_prefix}.level{level}.pk"
        with open(output_file, 'wb') as f:
            pickle.dump(codes, f)
        print(f"Saved level {level}: {len(codes)} codes → {output_file}")

    print(f"Tree saved to {output_prefix}.level*.pk")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python build_trees.py <hierarchy.csv> <seqs> <types> <output_prefix>")
        sys.exit(1)
    build_tree(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
