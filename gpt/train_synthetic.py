# gpt/train_synthetic.py
import torch
import torch.nn as nn
import numpy as np
import pickle
import os
from tqdm import tqdm
from model import GPTModel
from config import GPTConfig

# Load processed data
data = pickle.load(open("gpt/processed_synthetic.pkl", "rb"))
train_data = data['train']
val_data = data['val']
config = data['config']
pad_token = data['pad_token']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Train patients: {len(train_data)}, Val patients: {len(val_data)}")
print(f"Vocab size: {config.total_vocab_size}, Seq len sẽ dùng: 1024")

# Tạo fixed-length sequences
def prepare_sequences(records, max_len=1024):
    seqs = []
    for seq in records:
        if len(seq) > max_len:
            seq = seq[:max_len]
        # Pad đúng thành max_len + 1 (vì label là shift 1)
        padded = seq + [pad_token] * (max_len + 1 - len(seq))
        seqs.append(padded)
    return torch.tensor(seqs, dtype=torch.long)

# Chuẩn bị tensor một lần duy nhất (không cần tạo lại mỗi epoch)
train_tensor = prepare_sequences(train_data, max_len=1024).to(device)
val_tensor = prepare_sequences(val_data, max_len=1024).to(device)

model = GPTModel(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)

os.makedirs("gpt/result", exist_ok=True)
best_loss = float('inf')
accumulation_steps = 4

print("Start training...")

for epoch in range(1, config.epoch + 1):
    model.train()
    total_loss = 0.0

    # Shuffle chỉ index, không tạo lại tensor → nhanh hơn 10x
    indices = np.random.permutation(len(train_tensor))
    
    optimizer.zero_grad()
    for step, i in enumerate(range(0, len(train_tensor), config.batch_size)):
        batch_idx = indices[i:i + config.batch_size]
        batch = train_tensor[batch_idx]

        loss, _, _ = model(batch, ehr_labels=batch)
        loss = loss / accumulation_steps
        loss.backward()

        total_loss += loss.item() * accumulation_steps  # phục hồi lại loss thật

        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(range(0, len(train_tensor), config.batch_size)):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for i in range(0, len(val_tensor), config.batch_size):
            batch = val_tensor[i:i + config.batch_size]
            loss, _, _ = model(batch, ehr_labels=batch)
            val_loss += loss.item()

    val_loss /= (len(val_tensor) + config.batch_size - 1) // config.batch_size

    print(f"Epoch {epoch:3d} | Train Loss: {total_loss/len(train_tensor):.4f} | Val Loss: {val_loss:.4f}")

    if val_loss < best_loss:
        best_loss = val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'config': config,
            'vocab_data': data
        }, "gpt/result/best_synthetic_gpt.pth")
        print("    → New best model saved!")

print("Training completed! Best val loss:", best_loss)
