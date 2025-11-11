#07_evaluate
# =========================================================
# 📂 CẤU HÌNH ĐƯỜNG DẪN
# =========================================================
import os
import pickle
import numpy as np
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score

DATA_DIR = Path("/kaggle/input/downstream-data/mtg_downstream_data")
HYBRID_DIR = Path("/kaggle/working/hybrid_results")
MODEL_PATH = DATA_DIR / "pretrain_model.npz"
TREE_PATH = DATA_DIR / "tree_synth.types"
CODE_MAP_PATH = DATA_DIR / "code_map.pkl"

# =========================================================
# 🧩 LOAD DỮ LIỆU
# =========================================================
print("🔹 Loading hybrid data...")
seqs = pickle.load(open(HYBRID_DIR / "merged.seqs", "rb"))
labels = pickle.load(open(HYBRID_DIR / "merged.labels", "rb"))
print(f"📊 Tổng số bệnh nhân: {len(seqs)}")

# =========================================================
# 📖 LOAD CODE MAP (ID ↔ ICD9)
# =========================================================
if CODE_MAP_PATH.exists():
    code_map = pickle.load(open(CODE_MAP_PATH, "rb"))
    if isinstance(code_map, dict):
        id2code = {v: k for k, v in code_map.items()}
        print(f"✅ Loaded ICD9 mapping từ code_map.pkl ({len(id2code)} mã)")
    else:
        raise ValueError("❌ code_map.pkl không đúng định dạng.")
else:
    id2code = None
    print("⚠️ Không tìm thấy file code_map.pkl → chỉ hiển thị chỉ số index.")

def decode_codes(indices):
    """Chuyển danh sách index → mã ICD9 nếu có mapping"""
    if id2code is None:
        return [int(i) for i in indices]
    return [id2code.get(int(i), f"UNK_{i}") for i in indices]

# =========================================================
# 🧱 LOAD MÔ HÌNH GRAM (embedding + weights)
# =========================================================
print("🔹 Loading fine-tuned model weights...")
model_data = np.load(MODEL_PATH, allow_pickle=True)
print(f"✅ Keys trong model: {list(model_data.keys())}")

if "W_emb" in model_data:
    embedding = model_data["W_emb"]
elif "w" in model_data and "w_tilde" in model_data:
    embedding = (model_data["w"] + model_data["w_tilde"]) / 2.0
else:
    raise KeyError("❌ Không tìm thấy embedding trong pretrain_model.npz")

print(f"Embedding shape: {embedding.shape}")

# =========================================================
# 🧠 HÀM DỰ ĐOÁN
# =========================================================
def predict_next_visit(seq):
    if len(seq) == 0:
        return np.zeros(embedding.shape[0])
    last_visit = [idx for idx in seq[-1] if idx < embedding.shape[0]]
    if len(last_visit) == 0:
        visit_vec = embedding.mean(axis=0)
    else:
        visit_vec = embedding[last_visit].mean(axis=0)
    sim = embedding @ visit_vec
    return np.argmax(sim)

def predict_topk(seq, k=5):
    if len(seq) == 0:
        return []
    last_visit = [idx for idx in seq[-1] if idx < embedding.shape[0]]
    if len(last_visit) == 0:
        visit_vec = embedding.mean(axis=0)
    else:
        visit_vec = embedding[last_visit].mean(axis=0)
    sim = embedding @ visit_vec
    return np.argsort(sim)[-k:]

# =========================================================
# ⚙️ TÍNH TOP-5 ACCURACY
# =========================================================
topk_hits = 0
for seq, label in zip(seqs, labels):
    if len(seq) < 1:
        continue
    topk_pred = predict_topk(seq, k=5)
    true_labels = [l for l in label[0] if l < embedding.shape[0]]
    if any(l in topk_pred for l in true_labels):
        topk_hits += 1
topk_acc = topk_hits / len(seqs)
print(f"Top-5 Accuracy: {topk_acc:.4f}")

# =========================================================
# ⚙️ CHẠY DỰ ĐOÁN VÀ ĐÁNH GIÁ
# =========================================================
print("🚀 Predicting next diagnosis codes ...")

y_true, y_pred = [], []

for i, (seq, label) in enumerate(zip(seqs, labels)):
    if len(seq) < 1:
        continue
    pred_idx = predict_next_visit(seq)

    true_vec = np.zeros(embedding.shape[0])
    for l in label[0]:
        if l < embedding.shape[0]:
            true_vec[l] = 1

    pred_vec = np.zeros(embedding.shape[0])
    pred_vec[pred_idx] = 1

    y_true.append(true_vec)
    y_pred.append(pred_vec)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

prec = precision_score(y_true, y_pred, average="micro", zero_division=0)
rec = recall_score(y_true, y_pred, average="micro", zero_division=0)
f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)

# =========================================================
# 📈 IN KẾT QUẢ
# =========================================================
print("\n🎯 Evaluation Results (multi-label setting):")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-score:  {f1:.4f}")

# =========================================================
# 🔍 HIỂN THỊ VÍ DỤ DỰ ĐOÁN MÃ BỆNH
# =========================================================
print("\n📋 Ví dụ dự đoán bệnh tiếp theo:")

for i, seq in enumerate(seqs[:5]):  # In 5 bệnh nhân đầu tiên
    topk_pred = predict_topk(seq, k=5)
    last_visit = seq[-1] if len(seq) > 0 else []
    print(f"\n🩺 Bệnh nhân {i+1}:")
    print(f"  🔹 Mã bệnh lần khám gần nhất: {decode_codes(last_visit[:10])}{'...' if len(last_visit) > 10 else ''}")
    print(f"  🔮 Dự đoán top-5 mã bệnh lần khám tiếp theo: {decode_codes(topk_pred)}")

print("\n✅ Đánh giá hoàn tất! Model GRAM (fine-tuned) đã được kiểm tra trên dữ liệu hybrid.")
