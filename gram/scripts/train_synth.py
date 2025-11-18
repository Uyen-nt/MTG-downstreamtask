# gram/scripts/train_synth.py

import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score

from gram.model.gram import GRAM, load_tree, pad_batch

# ==========================
# PATH CONFIG
# ==========================

SEQ_FILE = "gram/data/synthetic_converted/synthetic_mapped.seqs"
TREE_PREFIX = "gram/data/mimic3_tree"      
BEST_MODEL_OUT = "gram/data/synth_train_best.pt"
LAST_MODEL_OUT = "gram/data/synth_train_last.pt"

# ==========================
# DATA SPLITTING & UTILITIES
# ==========================

def clean_seqs(seqs):
    clean = []
    for p in seqs:
        v = [x for x in p if len(x) > 0]
        if len(v) >= 2:
            clean.append(v)
    return clean

def build_labels(seqs):
    X, Y = [], []
    for p in seqs:
        X.append(p[:-1])
        Y.append(p[1:])
    return X, Y

def split_data(seqs, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42):
    """Chia dữ liệu thành train/val/test sets"""
    assert train_ratio + val_ratio + test_ratio == 1.0, "Tỷ lệ phải cộng thành 1"
    
    np.random.seed(random_seed)
    indices = np.random.permutation(len(seqs))
    
    train_size = int(len(seqs) * train_ratio)
    val_size = int(len(seqs) * val_ratio)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    train_seqs = [seqs[i] for i in train_indices]
    val_seqs = [seqs[i] for i in val_indices]
    test_seqs = [seqs[i] for i in test_indices]
    
    print(f"Data split: Train={len(train_seqs)}, Val={len(val_seqs)}, Test={len(test_seqs)}")
    return train_seqs, val_seqs, test_seqs

def evaluate_model(model, val_seqs, num_codes, num_classes, device):
    """Đánh giá model trên validation set"""
    model.eval()
    total_loss = 0
    all_true = []
    all_pred = []
    
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    
    with torch.no_grad():
        for i in range(0, len(val_seqs), 32):
            x_batch = [p[:-1] for p in val_seqs[i:i+32]]
            y_batch = [p[1:] for p in val_seqs[i:i+32]]
            
            if not x_batch:  # Skip empty batches
                continue
                
            x, _, mask, _ = pad_batch(x_batch, num_classes, num_codes, device)
            _, y, _, _ = pad_batch(y_batch, num_classes, num_codes, device)
            
            pred = model(x, mask)
            y_labels = y.argmax(dim=-1)
            
            loss_per_step = loss_fn(pred.permute(0, 2, 1), y_labels)
            loss_masked = loss_per_step * mask
            batch_loss = loss_masked.sum() / mask.sum()
            
            total_loss += batch_loss.item()
            
            # Collect predictions for last visit for metrics calculation
            for b in range(pred.size(1)):
                last_idx = mask[:, b].sum().int() - 1
                if last_idx >= 0:
                    last_pred = pred[last_idx, b].cpu().numpy()
                    last_true = y[last_idx, b].cpu().numpy()
                    
                    pred_top10 = np.argsort(-last_pred)[:10]
                    true_codes = np.where(last_true == 1)[0]
                    
                    all_true.append(true_codes)
                    all_pred.append(pred_top10)
    
    # Calculate metrics
    metrics = calculate_metrics(all_true, all_pred, num_codes)
    metrics['loss'] = total_loss / max(1, len(val_seqs) // 32)
    
    return metrics

def calculate_metrics(true_list, pred_list, num_codes):
    """Tính các metrics đánh giá"""
    if not true_list:
        return {'precision': 0, 'recall': 0, 'f1': 0, 'jaccard': 0, 'top5_acc': 0, 'top10_acc': 0}
    
    # Top-k accuracy
    top5_acc = top_k_accuracy(true_list, pred_list, k=5)
    top10_acc = top_k_accuracy(true_list, pred_list, k=10)
    
    # Binary metrics (micro-average)
    all_true_binary = []
    all_pred_binary = []
    
    for true_codes, pred_codes in zip(true_list, pred_list):
        true_binary = np.zeros(num_codes)
        pred_binary = np.zeros(num_codes)
        
        true_binary[list(true_codes)] = 1
        pred_binary[list(pred_codes)] = 1
        
        all_true_binary.append(true_binary)
        all_pred_binary.append(pred_binary)
    
    all_true_binary = np.array(all_true_binary)
    all_pred_binary = np.array(all_pred_binary)
    
    precision = precision_score(all_true_binary, all_pred_binary, average='micro', zero_division=0)
    recall = recall_score(all_true_binary, all_pred_binary, average='micro', zero_division=0)
    f1 = f1_score(all_true_binary, all_pred_binary, average='micro', zero_division=0)
    jaccard = jaccard_score(all_true_binary, all_pred_binary, average='micro')
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'jaccard': jaccard,
        'top5_acc': top5_acc,
        'top10_acc': top10_acc
    }

def top_k_accuracy(true_list, pred_list, k=10):
    """Tính top-k accuracy"""
    correct = 0
    total = 0
    
    for true_codes, pred_codes in zip(true_list, pred_list):
        if len(set(true_codes) & set(pred_codes[:k])) > 0:
            correct += 1
        total += 1
    
    return correct / total if total > 0 else 0

# ==========================
# TRAINING FUNCTION
# ==========================

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("===== TRAIN GRAM WITH SYNTHETIC DATA =====")

    # 1) LOAD AND SPLIT SEQS
    seqs = pickle.load(open(SEQ_FILE, "rb"))
    seqs = clean_seqs(seqs)
    
    # Split data
    train_seqs, val_seqs, test_seqs = split_data(seqs)
    
    # Save test set for final evaluation
    pickle.dump(test_seqs, open("gram/data/synthetic_test.seqs", "wb"))
    
    # Build labels for training
    X_train, Y_train = build_labels(train_seqs)

    # 2) Compute num_codes
    num_codes = max(max(max(v) for v in patient) for patient in seqs) + 1
    num_classes = num_codes
    print("num_codes =", num_codes)

    # 3) LOAD TREE (REAL MIMIC3)
    tree_leaves, tree_ancestors = load_tree(TREE_PREFIX, num_codes, device=device)

    # Compute max index in tree for embedding table size
    all_idx = []
    for L, A in zip(tree_leaves, tree_ancestors):
        all_idx.append(L.max().item())
        all_idx.append(A.max().item())

    types = pickle.load(open(f"{TREE_PREFIX}.types", "rb"))
    all_idx.append(max(types.values()))

    max_index_in_tree = max(all_idx)
    print("max_index_in_tree =", max_index_in_tree)

    # 4) CREATE MODEL
    model = GRAM(
        input_dim=num_codes,
        num_classes=num_classes,
        num_levels=len(tree_leaves),
        emb_dim=128,
        att_dim=128,
        hidden_dim=128,
        tree_leaves=tree_leaves,
        tree_ancestors=tree_ancestors,
        device=device,
        max_index_in_tree=max_index_in_tree
    ).to(device)

    # 5) TRAINING SETUP
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    
    # Early stopping
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    best_epoch = 0
    
    # Training history
    train_losses = []
    val_metrics_history = []
    
    print("\nStarting training...")
    print("Epoch | Train Loss | Val Loss | Top-5 Acc | Top-10 Acc | F1-Score")
    print("-" * 65)
    
    # 6) TRAIN LOOP
    for epoch in range(50):  # Tăng epoch vì có early stopping
        model.train()
        total_train_loss = 0
        batch_count = 0

        # Training phase
        for i in range(0, len(X_train), 32):
            x_batch = X_train[i : i+32]
            y_batch = Y_train[i : i+32]
            
            if not x_batch:  # Skip empty batches
                continue

            x, _, mask, lengths = pad_batch(x_batch, num_classes, num_codes, device)
            _, y, _, _ = pad_batch(y_batch, num_classes, num_codes, device)

            pred = model(x, mask)
            y_labels = y.argmax(dim=-1)
            
            loss_per_step = loss_fn(pred.permute(0, 2, 1), y_labels)
            loss_masked = loss_per_step * mask
            loss = loss_masked.sum() / mask.sum()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()
            batch_count += 1

        avg_train_loss = total_train_loss / batch_count if batch_count > 0 else 0
        train_losses.append(avg_train_loss)
        
        # Validation phase
        val_metrics = evaluate_model(model, val_seqs, num_codes, num_classes, device)
        val_metrics_history.append(val_metrics)
        
        # Print progress
        print(f"{epoch+1:5d} | {avg_train_loss:10.4f} | {val_metrics['loss']:8.4f} | "
              f"{val_metrics['top5_acc']:9.4f} | {val_metrics['top10_acc']:10.4f} | "
              f"{val_metrics['f1']:8.4f}")
        
        # Early stopping check
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), BEST_MODEL_OUT)
            print(f" → Saved NEW BEST model! (val_loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}. Best epoch: {best_epoch+1}")
                break

    # 7) FINAL EVALUATION
    print("\n" + "="*60)
    print("FINAL EVALUATION ON TEST SET")
    print("="*60)
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load(BEST_MODEL_OUT))
    test_metrics = evaluate_model(model, test_seqs, num_codes, num_classes, device)
    
    print("Test Set Metrics:")
    print(f"Loss:     {test_metrics['loss']:.4f}")
    print(f"Top-5 Acc: {test_metrics['top5_acc']:.4f}")
    print(f"Top-10 Acc: {test_metrics['top10_acc']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall:    {test_metrics['recall']:.4f}")
    print(f"F1-Score:  {test_metrics['f1']:.4f}")
    print(f"Jaccard:   {test_metrics['jaccard']:.4f}")
    
    # Save final model
    torch.save(model.state_dict(), LAST_MODEL_OUT)
    print(f"\nTraining finished!")
    print(f"Best model: {BEST_MODEL_OUT} (epoch {best_epoch+1})")
    print(f"Last model: {LAST_MODEL_OUT}")

if __name__ == "__main__":
    train()
