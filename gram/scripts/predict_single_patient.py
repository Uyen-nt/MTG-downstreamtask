# gram/scripts/predict_single_patient.py

import sys
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT)

import pickle
import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score

from gram.model.gram import GRAM, load_tree, pad_batch

MODEL = "gram/data/finetuned_best.pt"
TREE_PREFIX = "gram/data/mimic3_tree"
MIMIC_TYPES = "gram/data/mimic3_tree.types"

class GRAMPredictor:
    def __init__(self, model_path=MODEL):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.types = None
        self.idx2code = None
        self.num_codes = 0
        self.num_classes = 0
        self.tree_leaves = None
        self.tree_anc = None
        
        self.load_model_and_mapping(model_path)
    
    def load_model_and_mapping(self, model_path):
        """Load model và mapping cần thiết"""
        print("Loading model and mappings...")
        
        # Load types mapping
        self.types = pickle.load(open(MIMIC_TYPES, "rb"))
        self.num_codes = max(self.types.values()) + 1
        self.num_classes = self.num_codes
        self.idx2code = {v: k for k, v in self.types.items()}
        
        # Load tree
        self.tree_leaves, self.tree_anc = load_tree(TREE_PREFIX, self.num_codes, self.device)
        
        # Compute max index
        all_idx = []
        for L, A in zip(self.tree_leaves, self.tree_anc):
            all_idx.append(L.max().item())
            all_idx.append(A.max().item())
        all_idx.append(max(self.types.values()))
        max_index_in_tree = max(all_idx)
        
        # Load model
        self.model = GRAM(
            input_dim=self.num_codes,
            num_classes=self.num_classes,
            num_levels=len(self.tree_leaves),
            emb_dim=128,
            att_dim=128,
            hidden_dim=128,
            tree_leaves=self.tree_leaves,
            tree_ancestors=self.tree_anc,
            max_index_in_tree=max_index_in_tree,
            device=self.device,
        ).to(self.device)
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print("✅ Model loaded successfully!")
    
    def convert_icd_to_idx(self, icd_list):
        """Convert ICD strings to indices"""
        idx_list = []
        missing_codes = []
        
        for icd in icd_list:
            # Chuẩn hóa định dạng ICD
            standardized_icd = self.standardize_icd(icd)
            
            if standardized_icd in self.types:
                idx_list.append(self.types[standardized_icd])
            else:
                missing_codes.append(icd)
                print(f"⚠️  Mã '{icd}' (chuẩn hóa: '{standardized_icd}') không có trong từ điển → bỏ qua")
        
        return idx_list, missing_codes
    
    def standardize_icd(self, icd_str):
        """Chuẩn hóa định dạng ICD9 giống MIMIC3"""
        icd_str = str(icd_str).strip()
        
        # Nếu đã có format D_xxx.xx thì giữ nguyên
        if icd_str.startswith('D_'):
            return icd_str
        
        # Convert to standard ICD9 format
        if icd_str.startswith('E'):
            if len(icd_str) > 4:
                return f"D_{icd_str[:4]}.{icd_str[4:]}"
            else:
                return f"D_{icd_str}"
        else:
            if len(icd_str) > 3:
                return f"D_{icd_str[:3]}.{icd_str[3:]}"
            else:
                return f"D_{icd_str}"
    
    def predict_next_visit(self, history_visits, top_k=10):
        """Dự đoán visit tiếp theo từ lịch sử"""
        if len(history_visits) < 1:
            raise ValueError("Cần ít nhất 1 visit trong lịch sử")
        
        # Convert visits to indices
        history_idx = []
        all_missing = []
        
        for visit in history_visits:
            visit_idx, missing = self.convert_icd_to_idx(visit)
            history_idx.append(visit_idx)
            all_missing.extend(missing)
        
        # Prepare input data
        Xpad, _, mask, _ = pad_batch([history_idx], self.num_classes, self.num_codes, self.device)
        
        # Predict
        with torch.no_grad():
            pred = self.model(Xpad, mask).squeeze(1)
        
        last_pred = pred[-1].cpu().numpy()
        top_indices = np.argsort(-last_pred)[:top_k].tolist()
        
        # Convert back to ICD codes
        top_predictions = []
        for idx in top_indices:
            icd_code = self.idx2code.get(idx, f"UNKNOWN_{idx}")
            probability = float(last_pred[idx])
            top_predictions.append({
                'icd': icd_code.replace('D_', ''),
                'probability': probability,
                'index': idx
            })
        
        return top_predictions, all_missing
    
    def calculate_metrics(self, true_codes, predicted_codes, num_classes):
        """Tính các metrics đánh giá"""
        # Create binary vectors
        true_binary = np.zeros(num_classes)
        pred_binary = np.zeros(num_classes)
        
        true_indices, _ = self.convert_icd_to_idx(true_codes)
        pred_indices = [p['index'] for p in predicted_codes]
        
        true_binary[true_indices] = 1
        pred_binary[pred_indices[:len(true_indices)]] = 1  # Top-k prediction
        
        # Calculate metrics
        precision = precision_score(true_binary, pred_binary, zero_division=0)
        recall = recall_score(true_binary, pred_binary, zero_division=0)
        f1 = f1_score(true_binary, pred_binary, zero_division=0)
        jaccard = jaccard_score(true_binary, pred_binary)
        
        # Top-k accuracy
        hit = len(set(true_indices) & set(pred_indices))
        top_k_accuracy = hit / len(true_indices) if true_indices else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'jaccard': jaccard,
            'top_k_accuracy': top_k_accuracy,
            'hit_count': hit,
            'total_true': len(true_indices)
        }

def main():
    predictor = GRAMPredictor()
    
    print("=" * 60)
    print("🌟 GRAM MODEL - DỰ ĐOÁN BỆNH TIẾP THEO")
    print("=" * 60)
    print("\nHƯỚNG DẪN:")
    print("- Nhập các mã bệnh cho từng visit, cách nhau bằng dấu phẩy")
    print("- Mã bệnh có thể là ICD9 gốc (vd: 4019, 25000) hoặc đã chuẩn hóa")
    print("- Để kết thúc nhập visits, gõ 'done'")
    print("- Để thêm visit thực tế để đánh giá, gõ 'real'")
    print("- Để thoát, gõ 'exit'")
    print("\n" + "=" * 60)
    
    history_visits = []
    true_next_visit = None
    
    while True:
        if not history_visits:
            print("\n📝 NHẬP LỊCH SỬ BỆNH ÁN")
        
        visit_input = input(f"\nVisit {len(history_visits) + 1}: ").strip()
        
        if visit_input.lower() == 'done':
            if len(history_visits) < 1:
                print("❌ Cần ít nhất 1 visit để dự đoán!")
                continue
            break
        
        if visit_input.lower() == 'real' and history_visits:
            print("🔍 Nhập visit thực tế để đánh giá model:")
            real_input = input("Visit thực tế: ").strip()
            true_codes = [code.strip() for code in real_input.split(',')]
            true_next_visit = true_codes
            print("✅ Đã lưu visit thực tế để đánh giá")
            continue
        
        if visit_input.lower() == 'exit':
            return
        
        # Process visit codes
        codes = [code.strip() for code in visit_input.split(',')]
        history_visits.append(codes)
        print(f"✅ Đã thêm Visit {len(history_visits)}: {codes}")
    
    # Perform prediction
    print("\n" + "=" * 60)
    print("🤖 ĐANG DỰ ĐOÁN...")
    
    try:
        predictions, missing_codes = predictor.predict_next_visit(history_visits, top_k=15)
        
        # Display results
        print("\n" + "=" * 60)
        print("📊 KẾT QUẢ DỰ ĐOÁN")
        print("=" * 60)
        
        print("\n--- LỊCH SỬ ---")
        for i, visit in enumerate(history_visits, 1):
            print(f"Visit {i}: {visit}")
        
        print("\n--- DỰ ĐOÁN VISIT TIẾP THEO (Top-15) ---")
        print("Mã ICD9    | Xác suất   | Đánh giá")
        print("-" * 40)
        
        for i, pred in enumerate(predictions, 1):
            is_hit = ""
            if true_next_visit:
                # Check if this prediction is in true visit
                true_icds = [predictor.standardize_icd(code).replace('D_', '') for code in true_next_visit]
                if pred['icd'] in true_icds:
                    is_hit = "✅ ĐÚNG"
            
            print(f"{pred['icd']:10} | {pred['probability']:.4f}    | {is_hit}")
        
        # Calculate metrics if true visit is provided
        if true_next_visit:
            print("\n--- ĐÁNH GIÁ ĐỘ CHÍNH XÁC ---")
            metrics = predictor.calculate_metrics(true_next_visit, predictions, predictor.num_codes)
            
            print(f"Precision:    {metrics['precision']:.4f}")
            print(f"Recall:       {metrics['recall']:.4f}")
            print(f"F1-Score:     {metrics['f1_score']:.4f}")
            print(f"Jaccard:      {metrics['jaccard']:.4f}")
            print(f"Top-K Acc:    {metrics['top_k_accuracy']:.4f}")
            print(f"Số mã dự đoán đúng: {metrics['hit_count']}/{metrics['total_true']}")
            
            print("\n--- VISIT THỰC TẾ ---")
            print(f"{true_next_visit}")
        
        if missing_codes:
            print(f"\n⚠️  Cảnh báo: {len(missing_codes)} mã không có trong từ điển")
            print(f"   Các mã bị bỏ qua: {missing_codes}")
    
    except Exception as e:
        print(f"❌ Lỗi khi dự đoán: {str(e)}")

if __name__ == "__main__":
    main()
