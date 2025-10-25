# GRAM Downstream Task with MTGAN Synthetic Data

## Pipeline
1. `01_convert_synthetic.py` → Convert MTGAN output
2. `02_create_labels.py` → Next-visit labels
3. `03_build_tree.py` → ICD9 hierarchy
4. `04_pretrain.py` → Pre-train on synthetic
5. `05_finetune.py` → Fine-tune on real
6. `06_train_hybrid.py` → Real + k×Synthetic
7. `07_evaluate.py` → AUC, w-F1, Rare-F1

## Results
| Method | AUC | w-F1 | Rare-F1 |
|--------|-----|------|---------|
| Real only | xxx | xxx | xxx |
| Pretrain + Finetune | **xxx** | **xxx** | **xxx** |
