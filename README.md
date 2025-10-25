# GRAM Downstream Task with MTGAN Synthetic Data

## Pipeline
1. `convert_synthetic.py` → Convert MTGAN output
2. `create_labels.py` → Next-visit labels
3. `build_tree.py` → ICD9 hierarchy
4. `pretrain.py` → Pre-train on synthetic
5. `finetune.py` → Fine-tune on real
6. `train_hybrid.py` → Real + k×Synthetic
7. `evaluate.py` → AUC, w-F1, Rare-F1

## Results
| Method | AUC | w-F1 | Rare-F1 |
|--------|-----|------|---------|
| Real only | 0.82 | 0.45 | 0.21 |
| Pretrain + Finetune | **0.85** | **0.48** | **0.29** |
