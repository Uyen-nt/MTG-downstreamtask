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

# Tạo dataset dạng fixed length
def prepare_sequences(records, max_len=1000):
    seqs = []
    for seq in records:
        if len(seq) > max_len:
            seq = seq[:max_len]
        padded = seq + [pad_token] * (max_len + 1 - len(seq))
        seqs.append(padded)
    return torch.tensor(seqs, dtype=torch.long)

train_tensor = prepare_sequences(train_data)
val_tensor = prepare_sequences(val_data)

model = GPTModel(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)

os.makedirs("gpt/result", exist_ok=True)
best_loss = float('inf')

for epoch in range(1, config.epoch + 1):
    model.train()
    np.random.shuffle(train_data)
    train_tensor = prepare_sequences(train_data)
    total_loss = 0
    for i in range(0, len(train_tensor), config.batch_size):
        batch = train_tensor[i:i+config.batch_size].to(device)
        
        optimizer.zero_grad()
        loss, _, _ = model(batch, ehr_labels=batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for i in range(0, len(val_tensor), config.batch_size):
            batch = val_tensor[i:i+config.batch_size].to(device)
            loss, _, _ = model(batch, ehr_labels=batch)
            val_loss += loss.item()
    
    val_loss /= (len(val_tensor) // config.batch_size)
    print(f"Epoch {epoch} | Train Loss: {total_loss/len(train_tensor):.4f} | Val Loss: {val_loss:.4f}")
    
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save({
            'model': model.state_dict(),
            'config': config,
            'vocab': data
        }, "gpt/result/best_synthetic_gpt.pth")
        print("    → New best model saved!")

print("Training completed!")
