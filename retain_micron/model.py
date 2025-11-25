# retain_micron/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class RETAIN_Diagnosis(nn.Module):
    def __init__(self, n_codes, emb_size=256, dropout=0.5):
        super().__init__()
        self.emb = nn.Embedding(n_codes + 1, emb_size, padding_idx=n_codes)
        self.dropout = nn.Dropout(dropout)
        
        self.alpha_gru = nn.GRU(emb_size, emb_size, batch_first=True)
        self.beta_gru = nn.GRU(emb_size, emb_size, batch_first=True)
        
        self.alpha_linear = nn.Linear(emb_size, 1)
        self.beta_linear = nn.Linear(emb_size, emb_size)
        self.output = nn.Linear(emb_size, n_codes)

    def forward(self, visits, n_codes):
        # visits: list of list codes
        seq_lens = [len(v) for v in visits]
        max_len = max(seq_lens)
        
        padded = []
        for v in visits:
            if len(v) < max_len:
                v = v + [n_codes] * (max_len - len(v))
            padded.append(v)
        
        x = torch.LongTensor(padded).to(next(self.parameters()).device)  # (B, max_len)
        emb = self.dropout(self.emb(x))                                    # (B, max_len, dim)
        emb_sum = emb.sum(dim=1)                                           # (B, dim)

        _, h_alpha = self.alpha_gru(emb_sum.unsqueeze(0))
        _, h_beta = self.beta_gru(emb_sum.unsqueeze(0))
        
        alpha = self.alpha_linear(h_alpha.squeeze(0))   # (B, 1)
        beta = torch.tanh(self.beta_linear(h_beta.squeeze(0)))  # (B, dim)
        
        attn = F.softmax(alpha, dim=0)                  # (B, 1)
        context = (attn * beta * emb_sum).sum(dim=0)    # (dim,)
        
        logits = self.output(context)                   # (n_codes,)
        return logits
