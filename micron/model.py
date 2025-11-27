import torch
import torch.nn as nn
import torch.nn.functional as F

class MICRON_DX(nn.Module):
    def __init__(self, vocab_size, dcm, emb_dim=256, device=torch.device('cpu')):
        super().__init__()
        self.device = device
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.dropout = nn.Dropout(0.4)

        self.health_net = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
        )

        self.predictor = nn.Sequential(
            nn.Linear(emb_dim, emb_dim*2),
            nn.ReLU(),
            nn.Linear(emb_dim*2, vocab_size)
        )

        self.dcm = torch.tensor(dcm, dtype=torch.float32).to(device)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        for m in self.predictor:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, input):
        all_h = []

        for visit in input:
            if len(visit) == 0:
                continue
            emb = self.embedding(torch.LongTensor(visit).to(self.device))
            emb = emb.mean(dim=0, keepdim=True)
            h = self.health_net(emb)
            all_h.append(h)

        if len(all_h) == 0:
            raise ValueError("Patient has no valid visits")

        h_last = all_h[-1]
        h_prev = torch.stack(all_h[:-1]).mean(dim=0)

        h_res = h_last - h_prev
        logits = self.predictor(h_res)

        return logits
