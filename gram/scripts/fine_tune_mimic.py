# gram/scripts/fine_tune_mimic.py

import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT)

import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score

from gram.model.gram import GRAM, load_tree, pad_batch
from gram.scripts.train_synth import split_data, evaluate_model, calculate_metrics, top_k_accuracy

SYNTH_MODEL = "gram/data/synth_train_best.pt"
TREE_PREFIX = "gram/data/mimic3_tree"
MIMIC_FILE = "gram/data/mimic.seqs"

FT_BEST_OUT = "gram/data/finetuned_best.pt"
FT_LAST_OUT = "gram/data/finetuned_last.pt"

def clean_seqs(seqs):
    new = []
    for p in seqs:
        v = [x for x in p if len(x) > 0]
        if len(v) >= 2:
            new.append(v)
    return new

def build_labels(seqs):
    X, Y = [], []
    for p in seqs:
        X.append(p[:-1])
        Y.append(p[1:])
    return X, Y

def train_finetune():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("===== FINE-TUNE GRAM WITH MIMIC3 REAL DATA =====")

    # 1) LOAD AND SPLIT DATA
    seqs = pickle.load(open(MIMIC_FILE, "rb"))
    seqs = clean_seqs(seqs)
    
    # Split data
    train_seqs, val_seqs, test_seqs = split_data(seqs, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    
    # Save test set for final evaluation
    pickle.dump(test_seqs, open("gram/data/mimic_test.seqs", "wb"))
    
    # Build labels for training
    X_train, Y_train = build_labels(train_seqs)

    num_codes = max(max(max(v) for v in p) for p in seqs) + 1
    num_classes = num_codes

    # 2) LOAD TREE
    tree_leaves, tree_anc = load_tree(TREE_PREFIX, num_codes, device)

    all_idx = []
    for L, A in zip(tree_leaves, tree_anc): 
        all_idx.append(L.max().item())
        all_idx.append(A.max().item())
    types = pickle.load(open(f"{TREE_PREFIX}.types", "rb"))
    all_idx.append(max(types.values()))

    max_index = max(all_idx)

    # 3) CREATE AND LOAD PRETRAINED MODEL
    model = GRAM(
        input_dim=num_codes,
        num_classes=num_classes,
        num_levels=len(tree_leaves),
        emb_dim=128,
        att_dim=128,
        hidden_dim=128,
        tree_leaves=tree_leaves,
        tree_ancestors=tree_anc,
        max_index_in_tree=max_index,
        device=device
    ).to(device)

    # Load pretrained weights
    try:
        model.load_state_dict(torch.load(SYNTH_MODEL, map_location=device))
        print("✅ Loaded pretrained (synthetic) model successfully.")
    except Exception as e:
        print(f"❌ Error loading pretrained model: {e}")
        print("⚠️  Training from scratch...")

    # 4) FINE-TUNING SETUP
    opt = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)  # Lower LR for fine-tuning
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    
    # Early stopping
    best_val_loss = float('inf')
    patience = 7
    patience_counter = 0
    best_epoch = 0
    
    print("\nStarting fine-tuning...")
    print("Epoch | Train Loss | Val Loss | Top-5 Acc | Top-10 Acc | F1-Score")
    print("-" * 65)
    
    # 5) FINE-TUNING LOOP
    for epoch in range(30):
        model.train()
        total_train_loss = 0
        batch_count = 0

        # Training
        for i in range(0, len(X_train), 32):
            xb = X_train[i:i+32]
            yb = Y_train[i:i+32]
            
            if not xb:  # Skip empty batches
                continue

            xpad, _, mask, len_ = pad_batch(xb, num_classes, num_codes, device)
            _, ypad, _, _ = pad_batch(yb, num_classes, num_codes, device)

            pred = model(xpad, mask)
            ylab = ypad.argmax(dim=-1)
            
            loss_step = loss_fn(pred.permute(0, 2, 1), ylab)
            loss = (loss_step * mask).sum() / mask.sum()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_train_loss += loss.item()
            batch_count += 1

        avg_train_loss = total_train_loss / batch_count if batch_count > 0 else 0
        
        # Validation
        val_metrics = evaluate_model(model, val_seqs, num_codes, num_classes, device)
        
        # Print progress
        print(f"{epoch+1:5d} | {avg_train_loss:10.4f} | {val_metrics['loss']:8.4f} | "
              f"{val_metrics['top5_acc']:9.4f} | {val_metrics['top10_acc']:10.4f} | "
              f"{val_metrics['f1']:8.4f}")
        
        # Early stopping
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), FT_BEST_OUT)
            print(f" → Saved NEW BEST fine-tuned model! (val_loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}. Best epoch: {best_epoch+1}")
                break

    # 6) FINAL EVALUATION
    print("\n" + "="*60)
    print("FINAL EVALUATION ON MIMIC TEST SET")
    print("="*60)
    
    # Load best fine-tuned model
    model.load_state_dict(torch.load(FT_BEST_OUT))
    test_metrics = evaluate_model(model, test_seqs, num_codes, num_classes, device)
    
    print("MIMIC Test Set Metrics (After Fine-tuning):")
    print(f"Loss:     {test_metrics['loss']:.4f}")
    print(f"Top-5 Acc: {test_metrics['top5_acc']:.4f}")
    print(f"Top-10 Acc: {test_metrics['top10_acc']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall:    {test_metrics['recall']:.4f}")
    print(f"F1-Score:  {test_metrics['f1']:.4f}")
    print(f"Jaccard:   {test_metrics['jaccard']:.4f}")
    
    # Compare with pretrained model on same test set
    print("\n" + "="*60)
    print("COMPARISON: PRETRAINED vs FINE-TUNED")
    print("="*60)
    
    # Evaluate pretrained model on MIMIC test set
    pretrained_model = GRAM(
        input_dim=num_codes,
        num_classes=num_classes,
        num_levels=len(tree_leaves),
        emb_dim=128,
        att_dim=128,
        hidden_dim=128,
        tree_leaves=tree_leaves,
        tree_ancestors=tree_anc,
        max_index_in_tree=max_index,
        device=device
    ).to(device)
    
    pretrained_model.load_state_dict(torch.load(SYNTH_MODEL, map_location=device))
    pretrained_metrics = evaluate_model(pretrained_model, test_seqs, num_codes, num_classes, device)
    
    print(f"{'Metric':<12} | {'Pretrained':<10} | {'Fine-tuned':<10} | {'Improvement':<12}")
    print("-" * 55)
    print(f"{'Top-5 Acc':<12} | {pretrained_metrics['top5_acc']:10.4f} | {test_metrics['top5_acc']:10.4f} | {test_metrics['top5_acc'] - pretrained_metrics['top5_acc']:11.4f}")
    print(f"{'Top-10 Acc':<12} | {pretrained_metrics['top10_acc']:10.4f} | {test_metrics['top10_acc']:10.4f} | {test_metrics['top10_acc'] - pretrained_metrics['top10_acc']:11.4f}")
    print(f"{'F1-Score':<12} | {pretrained_metrics['f1']:10.4f} | {test_metrics['f1']:10.4f} | {test_metrics['f1'] - pretrained_metrics['f1']:11.4f}")
    print(f"{'Loss':<12} | {pretrained_metrics['loss']:10.4f} | {test_metrics['loss']:10.4f} | {pretrained_metrics['loss'] - test_metrics['loss']:11.4f}")
    
    # Save final model
    torch.save(model.state_dict(), FT_LAST_OUT)
    print(f"\nFine-tuning finished!")
    print(f"Best model: {FT_BEST_OUT}")
    print(f"Last model: {FT_LAST_OUT}")

if __name__ == "__main__":
    train_finetune()
