#07_evaluate
# =========================================================
# 🚀 Evaluate GRAM (fine-tuned) on real MIMIC-III for next-visit prediction
# =========================================================
import os
import pickle
import numpy as np
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score

# =========================================================
# 📂 CẤU HÌNH ĐƯỜNG DẪN
# =========================================================
PROJECT_ROOT = Path("/kaggle/working/MTG-downstreamtask")
GRAM_DIR = PROJECT_ROOT / "gram"
DATA_DIR = GRAM_DIR / "data"
RESULTS_DIR = GRAM_DIR / "results"

# ✅ DỮ LIỆU REAL MIMIC-III
REAL_SEQS = DATA_DIR / "tree_mimic3.seqs"
REAL_LABELS = DATA_DIR / "tree_mimic3.labels"

# ✅ MÔ HÌNH FINE-TUNED
MODEL_PATH = RESULTS_DIR / "finetune_real" / "pretrain_model.npz"

# ✅ CODE MAP (nếu có)
CODE_MAP_PATH = DATA_DIR / "tree_mimic3.types"

# =========================================================
# 🧩 LOAD DỮ LIỆU REAL
# =========================================================
print("🔹 Loading real MIMIC-III data ...")
seqs = pickle.load(open(REAL_SEQS, "rb"))
labels = pickle.load(open(REAL_LABELS, "rb"))
print(f"📊 Tổng số bệnh nhân: {len(seqs)}")

# =========================================================
# 📖 LOAD CODE MAP (ID ↔ ICD9)
# =========================================================
if CODE_MAP_PATH.exists():
    code_map = pickle.load(open(CODE_MAP_PATH, "rb"))
    if isinstance(code_map, dict):
        id2code = {v: k for k, v in code_map.items()}
        print(f"✅ Loaded ICD9 mapping từ tree_mimic3.types ({len(id2code)} mã)")
    else:
        id2code = None
        print("⚠️ File tree_mimic3.types không đúng định dạng dict.")
else:
    id2code = None
    print("⚠️ Không tìm thấy file tree_mimic3.types → chỉ hiển thị index.")

def decode_codes(indices):
    if id2code is None:
        return [int(i) for i in indices]
    return [id2code.get(int(i), f"UNK_{i}") for i in indices]

# =========================================================
# 🧱 LOAD MÔ HÌNH GRAM (embedding + weights)
# =========================================================
print("🔹 Loading fine-tuned GRAM weights ...")
model_data = np.load(MODEL_PATH, allow_pickle=True)
print(f"✅ Keys trong model: {list(model_data.keys())}")

if "W_emb" in model_data:
    embedding = model_data["W_emb"]
elif "w" in model_data and "w_tilde" in model_data:
    embedding = (model_data["w"] + model_data["w_tilde"]) / 2.0
else:
    raise KeyError("❌ Không tìm thấy embedding trong model file")

print(f"Embedding shape: {embedding.shape}")

# =========================================================
# 🧠 HÀM DỰ ĐOÁN
# =========================================================
def predict_next_visit(seq):
    """Trả về chỉ số mã bệnh dự đoán cao nhất cho lần khám kế tiếp."""
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
    """Trả về top-k mã bệnh có độ tương đồng cao nhất."""
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
valid_cases = 0

for seq, label in zip(seqs, labels):
    if len(seq) < 1 or len(label) < 1:
        continue
    topk_pred = predict_topk(seq, k=5)
    true_labels = [l for l in label[0] if l < embedding.shape[0]]
    if len(true_labels) == 0:
        continue
    valid_cases += 1
    if any(l in topk_pred for l in true_labels):
        topk_hits += 1

topk_acc = topk_hits / max(valid_cases, 1)
print(f"\n🎯 Top-5 Accuracy: {topk_acc:.4f} ({topk_hits}/{valid_cases})")

# =========================================================
# ⚙️ CHẠY DỰ ĐOÁN VÀ ĐÁNH GIÁ
# =========================================================
print("\n🚀 Predicting next diagnosis codes ...")

y_true, y_pred = [], []

for seq, label in zip(seqs, labels):
    if len(seq) < 1 or len(label) < 1:
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
print("\n📊 Evaluation Results (next-visit prediction on REAL MIMIC-III):")
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

print("\n✅ Đánh giá hoàn tất! Model GRAM (fine-tuned trên real) đã được kiểm tra.")
