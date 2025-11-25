# retain_micron/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class RETAIN_Diagnosis_Only(nn.Module):
    def __init__(self, n_diag_codes, emb_size=256, dropout=0.5):
        super().__init__()
        self.n_codes = n_diag_codes
        self.emb = nn.Embedding(n_diag_codes + 1, emb_size, padding_idx=n_diag_codes)
        self.dropout = nn.Dropout(dropout)
        
        self.alpha_gru = nn.GRU(emb_size, emb_size, batch_first=True)
        self.beta_gru  = nn.GRU(emb_size, emb_size, batch_first=True)
        self.alpha_lin = nn.Linear(emb_size, 1)
        self.beta_lin  = nn.Linear(emb_size, emb_size)
        self.output    = nn.Linear(emb_size, n_diag_codes)

    def forward(self, visits):
        # visits: List[List[List[int]]] → mỗi patient có nhiều visit, mỗi visit là list codes
        device = next(self.parameters()).device
        batch_size = len(visits)
        max_visits = max(len(p) for p in visits)

        # Tạo visit embedding
        visit_embs = []
        for patient in visits:
            p_embs = []
            for visit in patient:
                if not visit:
                    visit = [self.n_codes]
                codes = torch.LongTensor(visit).to(device)
                emb = self.dropout(self.emb(codes))
                v_emb = emb.sum(0)
                p_embs.append(v_emb)
            # Pad visit
            while len(p_embs) < max_visits:
                p_embs.append(torch.zeros(self.emb.embedding_dim, device=device))
            visit_embs.append(torch.stack(p_embs))
        
        x = torch.stack(visit_embs)  # (B, T, D)

        # RETAIN attention
        _, h_alpha = self.alpha_gru(x)
        _, h_beta  = self.beta_gru(x)
        
        alpha = self.alpha_lin(h_alpha.squeeze(0))           # (B, 1)
        beta  = torch.tanh(self.beta_lin(h_beta.squeeze(0))) # (B, D)
        
        attn = F.softmax(alpha, dim=0)
        context = (attn * beta * x.mean(dim=1)).sum(0)
        
        return self.output(context)
