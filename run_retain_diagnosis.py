import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import os
import numpy as np

from retain.utils import load_and_preprocess_synthetic
from retain.model import RETAIN_Single
from retain.dataset import EHRDataset, collate_fn
from retain.train import train_model
from retain.evaluate import print_final_evaluation

def debug_data_structure(data_path):
    data = np.load(data_path)
    x = data['x']
    lens = data['lens']
    
    print(f"=== DATA STRUCTURE DEBUG ===")
    print(f"x shape: {x.shape}")  # (n_patients, max_visits, n_codes)
    print(f"lens shape: {lens.shape}")
    print(f"Sample patient 0:")
    print(f"  - lens[0]: {lens[0]} (real visits)")
    
    # Kiểm tra patient đầu tiên
    for j in range(min(3, lens[0])):  # 3 visits đầu
        codes = np.where(x[0, j] == 1)[0]
        print(f"  - Visit {j}: {len(codes)} codes - {codes[:5]}...")
    
    # Kiểm tra patient cuối cùng  
    last_pid = len(lens) - 1
    print(f"Sample patient {last_pid}:")
    print(f"  - lens[{last_pid}]: {lens[last_pid]} (real visits)")
    for j in range(min(3, lens[last_pid])):
        codes = np.where(x[last_pid, j] == 1)[0]
        print(f"  - Visit {j}: {len(codes)} codes - {codes[:5]}...")
    
    return x, lens

def debug_data_statistics(seqs, labels, n_codes):
    print("\n===================================================")
    print("🔍 DEBUG DATA: CHECKING INPUT DATA STRUCTURE & LABELS")
    print("===================================================")

    print(f"👉 Samples: {len(seqs)}")
    
    # 1. Kiểm tra 3 sample đầu
    for i in range(3):
        print(f"\n--- Patient {i} ---")
        print(f"Số visit trong history: {len(seqs[i])}")
        for j, v in enumerate(seqs[i][:3]):
            print(f" Visit {j}: {len(v)} codes — {v[:5]}...")
        print(f"Label cuối: {len(labels[i])} codes — {labels[i][:10]}...")

    # 2. Thống kê số lượng mã bệnh xuất hiện trong label
    flat_label_counts = np.zeros(n_codes)
    empty_labels = 0

    for lb in labels:
        if len(lb) == 0:
            empty_labels += 1
            continue
        for code in lb:
            if code < n_codes:
                flat_label_counts[code] += 1

    print(f"\n💡 Label rỗng (len=0): {empty_labels} sample")
    print(f"💡 Mã bệnh chưa từng xuất hiện trong label: {(flat_label_counts==0).sum()} / {n_codes}")
    print(f"💡 Mã bệnh xuất hiện ít nhất 1 lần: {(flat_label_counts>0).sum()} / {n_codes}")

    print(f"\n🔝 Top 10 label phổ biến nhất:")
    top_codes = flat_label_counts.argsort()[-10:][::-1]
    for c in top_codes:
        print(f" Code {c} — {flat_label_counts[c]} lần")

    # 3. Trung bình số code/visit
    visit_code_counts = []
    for s in seqs:
        for v in s:
            visit_code_counts.append(len(v))

    print(f"\n📊 Số code trung bình / visit: {np.mean(visit_code_counts):.3f}")
    print(f"📊 Min–max code/visit: {np.min(visit_code_counts)} – {np.max(visit_code_counts)}")

    # 4. Trung bình số code trong label
    label_sizes = [len(lb) for lb in labels]
    print(f"\n📊 Trung bình số code trong label (ground truth final visit): {np.mean(label_sizes):.2f}")
    print(f"📊 Min–max code/label: {np.min(label_sizes)} – {np.max(label_sizes)}")

    print("===================================================")


def main():
    # Load data
    (train_seqs, train_labels), (test_seqs, test_labels), n_codes = load_and_preprocess_synthetic(
        data_path="data/result/synthetic_mimic3.npz"
    )
    debug_data_statistics(train_seqs, train_labels, n_codes)
    debug_data_statistics(test_seqs, test_labels, n_codes)

    print(f"Training samples: {len(train_seqs)}")
    print(f"Test samples: {len(test_seqs)}")
    
    train_dataset = EHRDataset(train_seqs, train_labels)
    val_dataset = EHRDataset(test_seqs, test_labels)  # Dùng test set làm validation

    collate = lambda batch: collate_fn(batch, n_codes=n_codes)

    # Tăng batch_size hoặc giữ nguyên, nhưng số batch sẽ tăng
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate)

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    model = RETAIN_Diagnosis(n_codes=n_codes, emb_size=256, dropout=0.5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

     # Train hoặc load model đã train
    model_path = "retain_micron/result/retain_best.pth"
    last_path = "retain_micron/result/retain_last.pth"
    
    # if os.path.exists(model_path):
    #     print("Loading best model...")
    #     model.load_state_dict(torch.load(model_path))
    #     model.eval()
    # elif os.path.exists(last_path):
    #     print("Best model not found, loading last epoch model...")
    #     model.load_state_dict(torch.load(last_path))
    #     model.eval()
    # else:
    #     print("Training new model...")
    #     train_model(model, train_loader, val_loader, epochs=20, save_path="retain_micron/result")
    #     # Sau khi train xong, load lại model đã lưu
    #     if os.path.exists(model_path):
    #         model.load_state_dict(torch.load(model_path))
    #         model.eval()
    #     elif os.path.exists(last_path):
    #         model.load_state_dict(torch.load(last_path))
    #         model.eval()

    train_model(model, train_loader, val_loader, epochs=10, save_path="retain_micron/result")
    # Sau khi train xong, load lại model đã lưu
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        model.eval()
    elif os.path.exists(last_path):
        model.load_state_dict(torch.load(last_path))
        model.eval()
    
    # Đánh giá toàn diện
    print("\n" + "="*70)
    print("🧪 COMPLETE MODEL EVALUATION")
    print("="*70)
    
    evaluation_results = print_final_evaluation(model, val_loader, n_codes)

if __name__ == "__main__":
    main()
