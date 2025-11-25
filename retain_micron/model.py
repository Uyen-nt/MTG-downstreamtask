# retain_micron/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class RETAIN_Diagnosis(nn.Module):
    def __init__(self, n_codes, emb_size=256, dropout=0.5):
        super().__init__()
        self.n_codes = n_codes
        self.emb_size = emb_size
        
        self.emb = nn.Embedding(n_codes + 1, emb_size, padding_idx=n_codes)
        self.dropout = nn.Dropout(dropout)
        
        self.alpha_gru = nn.GRU(emb_size, emb_size, batch_first=True)
        self.beta_gru  = nn.GRU(emb_size, emb_size, batch_first=True)
        
        self.alpha_lin = nn.Linear(emb_size, 1)
        self.beta_lin  = nn.Linear(emb_size, emb_size)
        self.output    = nn.Linear(emb_size, n_codes)

    def forward(self, visits_batch):
        """
        visits_batch: List[ List[List[int]] ] – batch bệnh nhân, mỗi bệnh nhân có nhiều visit
        """
        device = next(self.parameters()).device
        batch_size = len(visits_batch)
        
        all_contexts = []
        for visits in visits_batch:  # Duyệt từng bệnh nhân trong batch
            if not visits:
                visits = [[self.n_codes]]
                
            # Step 1: Tạo visit embedding
            visit_embs = []
            for visit in visits:
                if not visit:
                    visit = [self.n_codes]
                codes = torch.LongTensor(visit).to(device)
                emb = self.dropout(self.emb(codes))        # (n_codes_in_visit, D)
                v_emb = emb.sum(dim=0)                     # (D,)
                visit_embs.append(v_emb)
            
            visit_tensor = torch.stack(visit_embs)             # (T, D)
            
            # Step 2: RETAIN attention – CHUẨN GỐC PAPER
            g, _ = self.alpha_gru(visit_tensor.unsqueeze(0))   # (1, T, D)
            h, _ = self.beta_gru(visit_tensor.unsqueeze(0))
            
            alpha = F.softmax(self.alpha_lin(g.squeeze(0)), dim=0)  # (T, 1) ← softmax theo visit
            beta  = torch.tanh(self.beta_lin(h.squeeze(0)))         # (T, D)
            
            context = torch.sum(alpha * beta * visit_tensor, dim=0)  # (D,)
            all_contexts.append(context)
        
        # Step 3: Gom toàn batch
        context_batch = torch.stack(all_contexts)          # (B, D)
        logits = self.output(context_batch)                # (B, n_codes)
        return logits.squeeze(0) if batch_size == 1 else logits
