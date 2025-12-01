import torch
from torch.utils.data import DataLoader
from model import BertForMultiLabelPrediction
from model import BertConfig
from data_synth_loader import SyntheticBEHRTDataset
from optimizer import adam
import numpy as np
from sklearn.model_selection import train_test_split

# =======================
# 1) LOAD ORIGINAL DATA
# =======================
data = np.load("data/result/synthetic_mimic3.npz")
x = data["x"]
lens = data["lens"]
indices = np.arange(len(x))

# =======================
# 2) SPLIT 80 / 10 / 10
# =======================
train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=42)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

np.savez("data/result/synth_train.npz", x=x[train_idx], lens=lens[train_idx])
np.savez("data/result/synth_val.npz",   x=x[val_idx],   lens=lens[val_idx])
np.savez("data/result/synth_test.npz",  x=x[test_idx],  lens=lens[test_idx])

print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

# =======================
# 3) LOAD DATASETS
# =======================
train_data = SyntheticBEHRTDataset("data/result/synth_train.npz", max_len=512)
val_data   = SyntheticBEHRTDataset("data/result/synth_val.npz",   max_len=512)
test_data  = SyntheticBEHRTDataset("data/result/synth_test.npz",  max_len=512)

train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
val_loader   = DataLoader(val_data,   batch_size=8, shuffle=False)
test_loader  = DataLoader(test_data,  batch_size=8, shuffle=False)

# =======================
# 4) MODEL INIT
# =======================
config = {
    'vocab_size': 2871,
    'hidden_size': 288,
    'seg_vocab_size': 34,
    'age_vocab_size': 2,
    'max_position_embedding': 1500,
    'hidden_dropout_prob': 0.1,
    'num_hidden_layers': 4,
    'num_attention_heads': 8,
    'attention_probs_dropout_prob': 0.1,
    'intermediate_size': 512,
    'hidden_act': 'gelu',
    'initializer_range': 0.02,
}

feature_dict = {
    'word':True,
    'seg':True,
    'age':False,
    'position': True
}

model_config = BertConfig(config)
model = BertForMultiLabelPrediction(model_config, num_labels=config['vocab_size'], feature_dict=feature_dict)
optimizer = adam(model.named_parameters())

# =======================
# 5) TRAIN + VAL + EARLY STOP
# =======================
best_val_loss = float('inf')
patience = 5
wait = 0

for epoch in range(50):
    # ===== TRAIN =====
    model.train()
    total_loss = 0
    for batch in train_loader:
        visits, age_ids, seg_ids, pos_ids, attention_mask, labels = batch

        loss, logits = model(visits, age_ids, seg_ids, pos_ids, attention_mask, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    print(f"\n🟩 Epoch {epoch} TRAIN loss = {avg_train_loss:.4f}")

    # ===== VALIDATE =====
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            visits, age_ids, seg_ids, pos_ids, attention_mask, labels = batch
            loss, _ = model(visits, age_ids, seg_ids, pos_ids, attention_mask, labels)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"🟦 Epoch {epoch} VAL loss = {avg_val_loss:.4f}")

    # ===== CHECK BEST =====
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        wait = 0
        torch.save(model.state_dict(), "behrt/result/best_val_model.pt")
        print(f"🔥 Best model saved at epoch {epoch} — VAL loss improved to {best_val_loss:.4f}")
    else:
        wait += 1
        print(f"⏳ No improvement, patience {wait}/{patience}")
        if wait >= patience:
            print("⛔ EARLY STOP — validation không giảm nữa.")
            break

# =======================
# 6) FINAL TEST
# =======================
model.load_state_dict(torch.load("behrt/result/best_val_model.pt"))
model.eval()

test_loss = 0
with torch.no_grad():
    for batch in test_loader:
        visits, age_ids, seg_ids, pos_ids, attention_mask, labels = batch
        loss, _ = model(visits, age_ids, seg_ids, pos_ids, attention_mask, labels)
        test_loss += loss.item()

avg_test_loss = test_loss / len(test_loader)
print(f"\n🟨 FINAL TEST LOSS = {avg_test_loss:.4f}")
