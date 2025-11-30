import torch
from torch.utils.data import DataLoader
from behrt.model import BertForMultiLabelPrediction
from behrt.data_synth_loader import SyntheticBEHRTDataset
from behrt.optimizer import adam

synth_data = SyntheticBEHRTDataset("data/result/synthetic_mimic3.npz")
loader = DataLoader(synth_data, batch_size=8, shuffle=True, num_workers=2)

config = ...  # lấy config như BEHRT gốc
feature_dict = { 'word': True, 'seg': True, 'age': False, 'position': True }

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
