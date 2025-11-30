import torch
from torch.utils.data import DataLoader
from behrt.model import BertForMultiLabelPrediction
from behrt.data_synth_loader import SyntheticBEHRTDataset
from behrt.optimizer import adam

synth_data = SyntheticBEHRTDataset("data/result/synthetic_mimic3.npz")
loader = DataLoader(synth_data, batch_size=8, shuffle=True, num_workers=2)

config = {
    'vocab_size': 2871,                # PAD + CLS + 2869 codes
    'hidden_size': 288,
    'seg_vocab_size': 34,              # 34 possible visits
    'age_vocab_size': 2,               # age disabled
    'max_position_embedding': 1500,    # enough tokens
    'hidden_dropout_prob': 0.1,
    'num_hidden_layers': 6,
    'num_attention_heads': 12,
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

model = BertForMultiLabelPrediction(config, num_labels=config.vocab_size, feature_dict=feature_dict)
optimizer = adam(model.named_parameters())

model.train()
for e in range(10):
    for visits, age_ids, seg_ids, pos_ids, attention_mask in loader:
        
        labels = ...
        
        loss, logits = model(visits, age_ids, seg_ids, pos_ids, attention_mask, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("loss =", loss.item())
