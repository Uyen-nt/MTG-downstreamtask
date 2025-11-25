# retain_micron/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class RETAIN_Diagnosis(nn.Module):
    def __init__(self, n_codes, emb_size=256, dropout=0.5):
        super().__init__()
        self.n_codes = n_codes
        self.emb = nn.Embedding(n_codes + 1, emb_size, padding_idx=n_codes)
        self.dropout = nn.Dropout(dropout)
        self.alpha_gru = nn.GRU(emb_size, emb_size, batch_first=True)
        self.beta_gru = nn.GRU(emb_size, emb_size, batch_first=True)
        self.alpha_lin = nn.Linear(emb_size, 1)
        self.beta_lin = nn.Linear(emb_size, emb_size)
        self.output = nn.Linear(emb_size, n_codes)

    def forward(self, visits_batch):
        # visits_batch: list of list of list → [B, variable_seq_len, variable_codes_per_visit]
        device = next(self.parameters()).device
        
        # Tìm max number of visits trong batch
        max_visits = max(len(patient) for patient in visits_batch)
        
        batch_visit_embs = []
        for patient in visits_batch:
            visit_embs = []
            for visit in patient:
                if len(visit) == 0:
                    visit = [self.n_codes]  # padding code
                codes = torch.LongTensor(visit).to(device)
                emb = self.emb(codes)
                visit_emb = self.dropout(emb).sum(dim=0)  # sum → visit embedding
                visit_embs.append(visit_emb)
            
            # Pad số visit nếu cần
            while len(visit_embs) < max_visits:
                visit_embs.append(torch.zeros(self.emb.embedding_dim).to(device))
                
            batch_visit_embs.append(torch.stack(visit_embs))  # (seq_len, dim)
        
        visit_embs_tensor = torch.stack(batch_visit_embs)  # (B, max_visits, dim)
        
        # RETAIN core
        _, h_alpha = self.alpha_gru(visit_embs_tensor)   # h_alpha: (1, B, dim)
        _, h_beta = self.beta_gru(visit_embs_tensor)
        
        alpha = self.alpha_lin(h_alpha.squeeze(0))       # (B, 1)
        beta = torch.tanh(self.beta_lin(h_beta.squeeze(0)))  # (B, dim)
        
        attn = F.softmax(alpha, dim=0)                   # (B, 1) → attention over patients in batch
        context = (attn * beta * visit_embs_tensor.mean(dim=1)).sum(dim=0)  # weighted sum
        
        logits = self.output(context)                    # (n_codes,)
        return logits
