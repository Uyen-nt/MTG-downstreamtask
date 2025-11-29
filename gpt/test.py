# gpt/test.py
import os
import torch
import pickle
import random
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from gpt.model import GPTModel
from gpt.config import GPTConfig
import torch.nn.functional as F
import sys
import sys, os
sys.path.append(os.getcwd())
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent)) 

data_dir = "gpt/result"
save_dir = "gpt/result"
os.makedirs(save_dir, exist_ok=True)

SEED = 4
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
config = GPTConfig()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# SỬA ĐƯỜNG DẪN TẠI ĐÂY
train_ehr_dataset = pickle.load(open(f"{data_dir}/trainDataset.pkl", "rb"))
index_to_code = pickle.load(open(f"{data_dir}/indexToCode.pkl", "rb"))

# Thêm tên các chronic condition (để hiển thị đẹp hơn)
label_names = [
    "Chronic Condition: Alzheimer or related disorders or senile",
    "Chronic Condition: Heart Failure",
    "Chronic Condition: Chronic Kidney Disease",
    "Chronic Condition: Cancer",
    "Chronic Condition: Chronic Obstructive Pulmonary Disease",
    "Chronic Condition: Depression",
    "Chronic Condition: Diabetes",
    "Chronic Condition: Ischemic Heart Disease",
    "Chronic Condition: Osteoporosis",
    "Chronic Condition: rheumatoid arthritis and osteoarthritis (RA/OA)",
    "Chronic Condition: Stroke/transient Ischemic Attack"
]
for i, name in enumerate(label_names):
    index_to_code[config.code_vocab_size + i] = name

model = GPTModel(config).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

# SỬA ĐƯỜNG DẪN MODEL TẠI ĐÂY
model_path = f"{save_dir}/gpt_model_best.pt"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Không tìm thấy model tại {model_path}\nChạy train.py trước!")

checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model'])
model.eval()
print("Đã load model thành công!")

# === Các hàm giữ nguyên ===
def sample_sequence(model, length, context, batch_size=None, device='cuda', sample=True):
    context = torch.tensor(context, device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    prev = context
    ehr = context
    past = None
    with torch.no_grad():
        for _ in range(length):
            code_logits, past = model(prev, past=past)
            code_logits = code_logits[:, -1, :]
            log_probs = F.softmax(code_logits, dim=-1)
            if sample:
                prev = torch.multinomial(log_probs, num_samples=1)
            else:
                prev = torch.argmax(log_probs, dim=1)
            ehr = torch.cat((ehr, prev), dim=1)
            if all([config.code_vocab_size + config.label_vocab_size + 3 in ehr[i] for i in range(batch_size)]):
                break
    return ehr.cpu().numpy()

def convert_ehr(ehrs, index_to_code=None):
    ehr_outputs = []
    for ehr in ehrs:
        ehr_output = []
        visit_output = []
        labels_output = np.zeros(config.label_vocab_size)
        started_visits = False
        for code in ehr[1:]:
            if not started_visits:
                if code == config.code_vocab_size + config.label_vocab_size + 1:
                    started_visits = True
                elif config.code_vocab_size <= code < config.code_vocab_size + config.label_vocab_size:
                    labels_output[code - config.code_vocab_size] = 1
            else:
                if code < config.code_vocab_size:
                    if code not in visit_output:
                        visit_output.append(index_to_code.get(code, code))
                elif code == config.code_vocab_size + config.label_vocab_size + 2:
                    if visit_output:
                        ehr_output.append(visit_output)
                        visit_output = []
                elif code == config.code_vocab_size + config.label_vocab_size + 3:
                    break
        if visit_output:
            ehr_output.append(visit_output)
        labels = [index_to_code[config.code_vocab_size + i] for i in np.where(labels_output)[0]]
        ehr_outputs.append({'visits': ehr_output, 'labels': labels})
    return ehr_outputs

# === Generate synthetic data ===
print("Bắt đầu sinh dữ liệu synthetic...")
synthetic_ehr_dataset = []
stoken = [config.code_vocab_size + config.label_vocab_size]  # Start token

for i in tqdm(range(0, len(train_ehr_dataset), 2 * config.batch_size)):
    bs = min(2 * config.batch_size, len(train_ehr_dataset) - i)
    batch = sample_sequence(model, config.n_ctx, stoken, batch_size=bs, device=device, sample=True)
    batch = convert_ehr(batch, index_to_code)
    synthetic_ehr_dataset.extend(batch)

# LƯU KẾT QUẢ VÀO ĐÚNG THƯ MỤC
output_file = f"{save_dir}/gptDataset.pkl"
pickle.dump(synthetic_ehr_dataset, open(output_file, "wb"))
print(f"HOÀN TẤT! Đã sinh {len(synthetic_ehr_dataset)} bệnh án synthetic")
print(f"Lưu tại: {output_file}")
