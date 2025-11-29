# gpt/train.py
import os
import torch
import numpy as np
import random
import pickle
from tqdm import tqdm
from gpt.model import GPTModel  # chú ý: import đúng đường dẫn model.py
from gpt.config import GPTConfig

# ================= ĐƯỜNG DẪN CHO KAGGLE =================
data_dir = "/kaggle/working/gpt/result"
save_dir = "/kaggle/working/gpt/result"
os.makedirs(save_dir, exist_ok=True)

# ================= SEED & DEVICE =================
SEED = 4
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
print(f"Using device: {device}")

# ================= LOAD DATA =================
print("Loading processed datasets...")
orig_train_ehr_dataset = pickle.load(open(f"{data_dir}/trainDataset.pkl", "rb"))
orig_val_ehr_dataset   = pickle.load(open(f"{data_dir}/valDataset.pkl", "rb"))

config = GPTConfig()
print(f"Config: batch_size={config.batch_size}, n_ctx={config.n_ctx}, epochs={config.epoch}")

# ================= CHUYỂN ĐỔI SANG ĐỊNH DẠNG SEQUENCE (với token đặc biệt) =================
def convert_to_sequence(dataset):
    sequences = []
    PAD = config.total_vocab_size - 1
    START_RECORD = config.code_vocab_size + config.label_vocab_size
    END_LABELS   = START_RECORD + 1
    END_VISIT    = START_RECORD + 2
    END_RECORD   = START_RECORD + 3

    for patient in dataset:
        seq = [PAD] * config.n_ctx
        seq[0] = START_RECORD
        idx = 1

        # Add labels
        for label_idx in patient['labels'].nonzero()[0]:
            seq[idx] = config.code_vocab_size + label_idx
            idx += 1
        seq[idx] = END_LABELS
        idx += 1

        # Add visits
        for visit in patient['visits']:
            for code in visit:
                if idx < config.n_ctx:
                    seq[idx] = code
                    idx += 1
            if idx < config.n_ctx:
                seq[idx] = END_VISIT
                idx += 1

        # End record
        if idx < config.n_ctx:
            seq[idx] = END_RECORD

        sequences.append(seq)
    return sequences

print("Converting datasets to model input format...")
train_ehr_dataset = convert_to_sequence(orig_train_ehr_dataset)
val_ehr_dataset   = convert_to_sequence(orig_val_ehr_dataset)

print(f"Train samples: {len(train_ehr_dataset)} | Val samples: {len(val_ehr_dataset)}")

# ================= DATALOADER HELPER =================
def get_batch(loc, batch_size, mode="train"):
    if mode == "train":
        data = train_ehr_dataset
    else:
        data = val_ehr_dataset
    batch = data[loc:loc + batch_size]
    return torch.tensor(batch, dtype=torch.long).to(device)

def shuffle_training_data():
    np.random.shuffle(train_ehr_dataset)

# ================= MODEL & OPTIMIZER =================
model = GPTModel(config).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

# Load checkpoint nếu có (từ lần train trước trên Kaggle)
checkpoint_path = f"{save_dir}/gpt_model_best.pt"
if os.path.exists(checkpoint_path):
    print("Loading previous best model...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("Loaded!")

# ================= TRAINING LOOP =================
global_loss = float('inf')
print("Start training...")

for epoch in tqdm(range(config.epoch), desc="Epochs"):
    shuffle_training_data()
    for i in range(0, len(train_ehr_dataset), config.batch_size):
        model.train()
        batch = get_batch(i, config.batch_size, "train")

        optimizer.zero_grad()
        loss, _, _ = model(batch, ehr_labels=batch)  # self-supervised: predict next token
        loss.backward()
        optimizer.step()

        # Log
        if i % (100 * config.batch_size) == 0 and i > 0:
            print(f"Epoch {epoch}, Step {i}, Train Loss: {loss.item():.6f}")

        # Validation
        if i % (250 * config.batch_size) == 0 and i > 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for v_i in range(0, len(val_ehr_dataset), config.batch_size):
                    batch_val = get_batch(v_i, config.batch_size, "valid")
                    val_loss, _, _ = model(batch_val, ehr_labels=batch_val)
                    val_losses.append(val_loss.item())
            avg_val_loss = np.mean(val_losses)
            print(f"Epoch {epoch} | Validation Loss: {avg_val_loss:.6f}")

            # Save best model
            if avg_val_loss < global_loss:
                global_loss = avg_val_loss
                state = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': avg_val_loss
                }
                torch.save(state, f"{save_dir}/gpt_model_best.pt")
                print("NEW BEST MODEL SAVED!")

print("TRAINING HOÀN TẤT!")
print(f"Best model lưu tại: {save_dir}/gpt_model_best.pt")
