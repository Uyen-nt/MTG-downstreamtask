# gram/scripts/03_build_tree_synth.py
import pickle, os
from pathlib import Path
import numpy as np

GRAM_DATA_DIR = Path("gram/data")
SYNTH_TYPES = GRAM_DATA_DIR / "synth_mimic3.types"
TREE_PREFIX  = GRAM_DATA_DIR / "tree_synth"

os.makedirs(GRAM_DATA_DIR, exist_ok=True)

def main():
    types = pickle.load(open(SYNTH_TYPES, "rb"))  # dict: code_str -> id  (hoặc id -> code_str)

    # Lấy tập id mã (0..V-1). Dù types là map nào, ta cần danh sách id liên tục.
    if all(isinstance(k, str) for k in types.keys()):
        ids = sorted(set(types.values()))
    else:
        ids = sorted(set(types.keys()))
    V = max(ids) + 1          # code id hợp lệ: 0..V-1

    # ❌ KHÔNG tạo ROOT = V nữa
    # ✅ Mỗi code sẽ có 5 ancestor đều = chính nó
    levels = {}
    for L in [5, 4, 3, 2, 1]:
        anc = np.zeros((V, 5), dtype=np.int32)
        for cid in range(V):
            anc[cid, :] = cid   # [cid, cid, cid, cid, cid]
        levels[L] = anc

    # Lưu .level#.pk dưới dạng dict: {cid: [anc_0,...,anc_4]}
    for L in [5, 4, 3, 2, 1]:
        tree_map = {cid: levels[L][cid].tolist() for cid in range(V)}
        with open(f"{TREE_PREFIX}.level{L}.pk", "wb") as f:
            pickle.dump(tree_map, f, -1)

    # Lưu .types: để gram.py đọc rootCode nhưng ta set root nằm trong [0..V-1]
    with open(f"{TREE_PREFIX}.types", "wb") as f:
        pickle.dump({"A_ROOT": V-1}, f, -1)

    print(f"✓ Built dummy tree for synth: {TREE_PREFIX}.level[1-5].pk and .types (V={V})")

if __name__ == "__main__":
    main()
