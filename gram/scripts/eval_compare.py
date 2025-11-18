# gram/scripts/eval_compare.py

import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT)

import pickle
import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score

from gram.model.gram import GRAM, load_tree, pad_batch
from gram.scripts.train_synth import evaluate_model, calculate_metrics, top_k_accuracy

# ===============================
# CONFIG
# ===============================
TREE_PREFIX = "gram/data/mimic3_tree"
TEST_FILE = "gram/data/mimic_test.seqs"  # Use the saved test set

SYNTH_MODEL = "gram/data/synth_train_best.pt"
FINETUNE_MODEL = "gram/data/finetuned_best.pt"

# ===============================
# COMPREHENSIVE EVALUATION
# ===============================

def load_test_data():
    """Load test data for evaluation"""
    try:
        test_seqs = pickle.load(open(TEST_FILE, "rb"))
        print(f"Loaded test set: {len(test_seqs)} patients")
        return test_seqs
    except:
        print("Test set not found, using full dataset...")
        full_seqs = pickle.load(open("gram/data/mimic.seqs", "rb"))
        # Clean and take subset for testing
        clean_seqs = []
        for p in full_seqs:
            v = [x for x in p if len(x) > 0]
            if len(v) >= 2:
                clean_seqs.append(v)
        return clean_seqs[:1000]  # Use subset for faster evaluation

def evaluate_all_models():
    """Evaluate both pretrained and fine-tuned models"""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("===== COMPREHENSIVE MODEL EVALUATION =====")
    
    # Load test data
    test_seqs = load_test_data()
    
    # Compute num_codes
    num_codes = max(max(max(v) for v in p) for p in test_seqs) + 1
    
    # Load tree
    tree_leaves, tree_anc = load_tree(TREE_PREFIX, num_codes, device)
    
    all_idx = []
    for L, A in zip(tree_leaves, tree_anc):
        all_idx.append(L.max().item())
        all_idx.append(A.max().item())
    types = pickle.load(open(f"{TREE_PREFIX}.types", "rb"))
    all_idx.append(max(types.values()))
    max_index_tree = max(all_idx)
    
    print(f"Number of codes: {num_codes}")
    print(f"Test patients: {len(test_seqs)}")
    
    # Evaluate both models
    results = {}
    
    for model_name, model_path in [("Synthetic Pretrained", SYNTH_MODEL), 
                                   ("Fine-tuned MIMIC", FINETUNE_MODEL)]:
        
        print(f"\n{'='*50}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*50}")
        
        # Create model
        model = GRAM(
            input_dim=num_codes,
            num_classes=num_codes,
            num_levels=len(tree_leaves),
            emb_dim=128,
            att_dim=128,
            hidden_dim=128,
            tree_leaves=tree_leaves,
            tree_ancestors=tree_anc,
            max_index_in_tree=max_index_tree,
            device=device,
        ).to(device)
        
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading {model_name}: {e}")
            continue
        
        # Evaluate
        metrics = evaluate_model(model, test_seqs, num_codes, num_codes, device)
        results[model_name] = metrics
        
        # Print results
        print(f"Loss:          {metrics['loss']:.4f}")
        print(f"Top-5 Accuracy: {metrics['top5_acc']:.4f}")
        print(f"Top-10 Accuracy: {metrics['top10_acc']:.4f}")
        print(f"Precision:     {metrics['precision']:.4f}")
        print(f"Recall:        {metrics['recall']:.4f}")
        print(f"F1-Score:      {metrics['f1']:.4f}")
        print(f"Jaccard:       {metrics['jaccard']:.4f}")
    
    # Print comparison table
    print("\n" + "="*80)
    print("FINAL COMPARISON RESULTS")
    print("="*80)
    
    if len(results) == 2:
        synth_metrics = results["Synthetic Pretrained"]
        ft_metrics = results["Fine-tuned MIMIC"]
        
        print(f"{'Metric':<15} | {'Synthetic Pretrain':<20} | {'Fine-tuned MIMIC':<20} | {'Improvement':<15}")
        print("-" * 85)
        
        metrics_list = [
            ('Top-5 Acc', synth_metrics['top5_acc'], ft_metrics['top5_acc']),
            ('Top-10 Acc', synth_metrics['top10_acc'], ft_metrics['top10_acc']),
            ('Precision', synth_metrics['precision'], ft_metrics['precision']),
            ('Recall', synth_metrics['recall'], ft_metrics['recall']),
            ('F1-Score', synth_metrics['f1'], ft_metrics['f1']),
            ('Jaccard', synth_metrics['jaccard'], ft_metrics['jaccard']),
            ('Loss', synth_metrics['loss'], ft_metrics['loss']),
        ]
        
        for name, synth_val, ft_val in metrics_list:
            if name == 'Loss':
                improvement = synth_val - ft_val  # Lower loss is better
            else:
                improvement = ft_val - synth_val  # Higher is better
            
            print(f"{name:<15} | {synth_val:<20.4f} | {ft_val:<20.4f} | {improvement:>14.4f}")
        
        # Calculate overall improvement
        key_metrics = ['top5_acc', 'top10_acc', 'f1']
        avg_improvement = np.mean([ft_metrics[m] - synth_metrics[m] for m in key_metrics])
        print(f"\n📈 Average improvement on key metrics: {avg_improvement:.4f}")
    
    print("="*80)

if __name__ == "__main__":
    evaluate_all_models()
