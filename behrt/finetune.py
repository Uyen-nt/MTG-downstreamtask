import torch
from torch.utils.data import DataLoader
from model import BertForMultiLabelPrediction, BertConfig
from data_mimic_loader import RealBEHRTDataset
from optimizer import adam
import numpy as np

# ============================
# 1) LOAD MIMIC REAL DATA
# ============================
data = np.load("data/result/mimic_train.npz")
val_data_np = np.load("data/result/mimic_val.npz")
test_data_np = np.load("data/result/mimic_test.npz")

train_data = RealBEHRTDataset("data/result/mimic_train.npz", max_len=512)
val_data   = RealBEHRTDataset("data/result/mimic_val.npz",   max_len=512)
test_data  = RealBEHRTDataset("data/result/mimic_test.npz",  max_len=512)

train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
val_loader   = DataLoader(val_data,   batch_size=8, shuffle=False)
test_loader  = DataLoader(test_data,  batch_size=8, shuffle=False)

print("✔ Loaded mimic fine-tune datasets")

# ============================
# 2) LOAD MODEL PRETRAINED ON SYNTHETIC
# ============================
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

print("🔹 Loading pretrained weights from synthetic ...")
model.load_state_dict(torch.load("behrt/result/best_val_model.pt", map_location="cpu"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("✔ Model loaded on", device)

# ============================
# 3) FREEZE LOWER LAYERS
# ============================
for name, param in model.named_parameters():
    if ("embeddings" in name) or ("encoder.layer.0" in name) or ("encoder.layer.1" in name):
        param.requires_grad = False
        # print("❄ FREEZE", name)
    else:
        param.requires_grad = True

print("✔ Finished freezing early layers")

# ============================
# 4) SET OPTIMIZER — ONLY TRAIN HOT LAYERS
# ============================
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = adam(trainable_params)

# ============================
# 5) TRAIN + VAL
# ============================
best_val_loss = float('inf')
patience = 5
wait = 0

for epoch in range(20):
    # ===== TRAIN =====
    model.train()
    total_loss = 0
    for batch in train_loader:
        visits, age_ids, seg_ids, pos_ids, attention_mask, labels = batch
        visits, age_ids, seg_ids, pos_ids, attention_mask, labels = \
            visits.to(device), age_ids.to(device), seg_ids.to(device), pos_ids.to(device), attention_mask.to(device), labels.to(device)

        loss, logits = model(visits, age_ids, seg_ids, pos_ids, attention_mask, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    print(f"\n🟩 Fine-tune Epoch {epoch} TRAIN loss = {avg_train_loss:.4f}")

    # ===== VALIDATE =====
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            visits, age_ids, seg_ids, pos_ids, attention_mask, labels = batch
            visits, age_ids, seg_ids, pos_ids, attention_mask, labels = \
                visits.to(device), age_ids.to(device), seg_ids.to(device), pos_ids.to(device), attention_mask.to(device), labels.to(device)

            loss, _ = model(visits, age_ids, seg_ids, pos_ids, attention_mask, labels)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"🟦 Fine-tune Epoch {epoch} VAL loss = {avg_val_loss:.4f}")

    # ===== SAVE BEST =====
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        wait = 0
        torch.save(model.state_dict(), "behrt/result/best_val_model_finetune.pt")
        print(f"🔥 BEST FINE-TUNE SAVED — {best_val_loss:.4f}")
    else:
        wait += 1
        if wait >= patience:
            print("⛔ STOP — NO MORE IMPROVEMENT")
            break

# ============================
# 6) FINAL TEST
# ============================
model.load_state_dict(torch.load("behrt/result/best_val_model_finetune.pt"))
model.eval()

test_loss = 0
with torch.no_grad():
    for batch in test_loader:
        visits, age_ids, seg_ids, pos_ids, attention_mask, labels = batch
        visits, age_ids, seg_ids, pos_ids, attention_mask, labels = \
            visits.to(device), age_ids.to(device), seg_ids.to(device), pos_ids.to(device), attention_mask.to(device), labels.to(device)
        loss, _ = model(visits, age_ids, seg_ids, pos_ids, attention_mask, labels)
        test_loss += loss.item()

print(f"\n🟨 FINAL TEST LOSS (FINE-TUNE) = {test_loss/ len(test_loader):.4f}")
