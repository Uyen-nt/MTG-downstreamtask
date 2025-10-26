# gram/scripts/03b_build_tree_synth.py
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
    V = max(ids) + 1

    # ROOT đặt là V (ngay sau vocab)
    ROOT = V
    # Mỗi code có 5 ancestor levels: [code, ROOT, ROOT, ROOT, ROOT]
    # Tương ứng level5 .. level1
    levels = {}
    for L in [5,4,3,2,1]:
        anc = np.zeros((V, 5), dtype=np.int32)
        for cid in range(V):
            anc[cid, 0] = cid
            anc[cid, 1:] = ROOT
        levels[L] = anc

    # leaves: [[cid]*5] để match shape với ancestors
    leaves = {}
    for L in [5,4,3,2,1]:
        anc = levels[L]
        anc_size = anc.shape[1]
        leaf = np.array([[cid]*anc_size for cid in range(V)], dtype=np.int32)
        leaves[L] = leaf

    # Lưu .level#.pk dưới dạng dict: {cid: [anc_0,...,anc_4]}
    for L in [5,4,3,2,1]:
        tree_map = {cid: levels[L][cid].tolist() for cid in range(V)}
        with open(f"{TREE_PREFIX}.level{L}.pk", "wb") as f:
            pickle.dump(tree_map, f, -1)

    # Lưu .types để gram.py lấy root theo cách cũ
    with open(f"{TREE_PREFIX}.types", "wb") as f:
        pickle.dump({"A_ROOT": ROOT}, f, -1)

    print(f"✓ Built trivial tree for synth: {TREE_PREFIX}.level[1-5].pk and .types (ROOT={ROOT}, V={V})")

if __name__ == "__main__":
    main()
