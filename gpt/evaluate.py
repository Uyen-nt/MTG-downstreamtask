# gpt/evaluate.py

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools
from scipy.stats import entropy
from sklearn.metrics import r2_score

# ================= ĐƯỜNG DẪN KAGGLE =================
base_dir = "/kaggle/working/gpt/result"
os.makedirs(f"{base_dir}/gpt_jsd_evaluation", exist_ok=True)
os.makedirs(f"{base_dir}/gpt_jsd_evaluation/plots", exist_ok=True)

# Load dữ liệu
real_data = pickle.load(open(f"{base_dir}/trainDataset.pkl", "rb"))
gpt_data  = pickle.load(open(f"{base_dir}/gptDataset.pkl", "rb"))
gpt_data = [p for p in gpt_data if len(p['visits']) > 0]

print(f"Real patients : {len(real_data):,}")
print(f"GPT patients  : {len(gpt_data):,}")

# ================= HÀM TÍNH PHÂN PHỐI =================
def get_distribution(dataset, mode="record_code"):
    counts = {}
    total = 0

    for patient in dataset:
        seen_codes = set()
        seen_bigrams = set()
        prev_visit = None

        for visit in patient['visits']:
            if mode == "record_code":
                seen_codes.update(visit)
            elif mode == "visit_code":
                for c in visit:
                    counts[c] = counts.get(c, 0) + 1
                    total += 1
            elif mode == "record_bigram":
                for pair in itertools.combinations(sorted(visit), 2):
                    seen_bigrams.add(pair)
            elif mode == "visit_bigram":
                for pair in itertools.combinations(sorted(visit), 2):
                    counts[pair] = counts.get(pair, 0) + 1
                    total += 1
            elif mode == "sequential":
                if prev_visit is not None:
                    for c1 in prev_visit:
                        for c2 in visit:
                            key = (c1, c2)
                            counts[key] = counts.get(key, 0) + 1
                            total += 1
            prev_visit = visit

        if mode == "record_code":
            for c in seen_codes:
                counts[c] = counts.get(c, 0) + 1
                total += 1
        elif mode == "record_bigram":
            for b in seen_bigrams:
                counts[b] = counts.get(b, 0) + 1
                total += 1

    return counts, total

# ================= TÍNH JSD & NDKL =================
def jensen_shannon_divergence(p, q):
    p = np.array(p)
    q = np.array(q)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return 0.5 * entropy(p, m) + 0.5 * entropy(q, m)

def normalized_kl(p, q):
    p = np.array(p) + 1e-12
    q = np.array(q) + 1e-12
    p = p / p.sum()
    q = q / q.sum()
    return entropy(p, q) / np.log(len(p))  # normalized

# ================= DANH SÁCH METRIC =================
metrics = [
    ("Record-level Code Frequency",        "record_code"),
    ("Visit-level Code Frequency",         "visit_code"),
    ("Record-level Bigram (co-occurrence)","record_bigram"),
    ("Visit-level Bigram (co-occurrence)", "visit_bigram"),
    ("Sequential Visit Bigram (temporal)", "sequential")
]

results = []

print("\nĐang tính JSD và NDKL...")
for name, mode in metrics:
    print(f"   → {name}...", end=" ")

    real_counts, real_total = get_distribution(real_data, mode)
    gpt_counts,  gpt_total  = get_distribution(gpt_data,  mode)

    all_keys = set(real_counts.keys()) | set(gpt_counts.keys())
    real_vec = np.array([real_counts.get(k, 0) for k in all_keys])
    gpt_vec  = np.array([gpt_counts.get(k, 0)  for k in all_keys])

    jsd = jensen_shannon_divergence(real_vec, gpt_vec)
    ndkl = normalized_kl(real_vec, gpt_vec)

    # R² nếu có ít nhất 10 điểm chung
    common_keys = set(real_counts.keys()) & set(gpt_counts.keys())
    if len(common_keys) > 10:
        r_real = np.array([real_counts[k] for k in common_keys])
        r_gpt  = np.array([gpt_counts[k]  for k in common_keys])
        r2 = r2_score(r_real, r_gpt)
    else:
        r2 = np.nan

    results.append({
        "metric": name,
        "JSD": jsd,
        "NDKL": ndkl,
        "R²": r2
    })
    print(f"JSD = {jsd:.4f}, NDKL = {ndkl:.4f}, R² = {r2:.4f}" if not np.isnan(r2) else f"JSD = {jsd:.4f}, NDKL = {ndkl:.4f}")

# ================= IN KẾT QUẢ ĐẸP =================
print("\n" + "="*80)
print("            KẾT QUẢ ĐÁNH GIÁ GPT SYNTHETIC DATA (vs MIMIC-III)")
print("="*80)
print(f"{'Metric':<45} {'JSD ↓':<10} {'NDKL ↓':<10} {'R² ↑':<8}")
print("-"*80)
for r in results:
    r2_str = f"{r['R²']:.4f}" if not np.isnan(r['R²']) else "N/A"
    print(f"{r['metric']:<45} {r['JSD']:.4f}     {r['NDKL']:.4f}     {r2_str}")
print("-"*80)

# ================= VẼ BIỂU ĐỒ (tùy chọn) =================
for r, (name, mode) in zip(results, metrics):
    if np.isnan(r['R²']): continue
    real_counts, _ = get_distribution(real_data, mode)
    gpt_counts,  _ = get_distribution(gpt_data,  mode)
    common = set(real_counts.keys()) & set(gpt_counts.keys())
    if len(common) < 10: continue

    x = [real_counts[k] for k in common]
    y = [gpt_counts[k]  for k in common]

    plt.figure(figsize=(6,5))
    plt.scatter(x, y, alpha=0.6, s=15)
    maxv = max(max(x), max(y)) * 1.1
    plt.plot([0, maxv], [0, maxv], 'r--')
    plt.xlim(0, maxv); plt.ylim(0, maxv)
    plt.xlabel("Real Frequency")
    plt.ylabel("GPT Frequency")
    plt.title(f"{name}\nJSD={r['JSD']:.4f} | NDKL={r['NDKL']:.4f} | R²={r['R²']:.4f}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
    plt.savefig(f"{base_dir}/gpt_jsd_evaluation/plots/{safe_name}.png", dpi=200)
    plt.close()

print(f"\nHOÀN TẤT! Biểu đồ lưu tại: {base_dir}/gpt_jsd_evaluation/plots/")
