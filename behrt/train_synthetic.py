import torch
from torch.utils.data import DataLoader
from model import BertForMultiLabelPrediction
from model import BertConfig
from data_synth_loader import SyntheticBEHRTDataset
from optimizer import adam

synth_data = SyntheticBEHRTDataset("data/result/synthetic_mimic3.npz", max_len=512)
loader = DataLoader(synth_data, batch_size=8, shuffle=True, num_workers=2)

config = {
    'vocab_size': 2871,                # PAD + CLS + 2869 codes
    'hidden_size': 288,
    'seg_vocab_size': 34,              # 34 possible visits
    'age_vocab_size': 2,               # age disabled
    'max_position_embedding': 1500,    # enough tokens
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

best_loss = float('inf')

model.train()
for epoch in range(10):

    total_loss = 0
    for step, batch in enumerate(loader):

        visits, age_ids, seg_ids, pos_ids, attention_mask, labels = batch

        loss, logits = model(visits, age_ids, seg_ids, pos_ids, attention_mask, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if step % 20 == 0:
            print(f"Epoch [{epoch+1}/10], Step [{step}/{len(loader)}], Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1} avg loss = {avg_loss:.4f}")

    # ===== CHECK IF BEST =====
    if avg_loss < best_loss:
        best_loss = avg_loss
        save_path = "behrt/result/best_behrt_synth.pt"
        
        torch.save({
            "epoch": epoch+1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "feature_dict": feature_dict,
            "best_loss": best_loss,
        }, save_path)

        print(f"🔥 New best model saved! Loss improved to {best_loss:.4f} at epoch {epoch+1}")

