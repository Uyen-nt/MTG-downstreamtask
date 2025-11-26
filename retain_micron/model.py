# retain_micron/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
        device = next(self.parameters()).device
        all_contexts = []
    
        for visits in visits_batch:
            if not visits or len(visits) == 0:
                visits = [[self.n_codes]]
    
            # === Chuẩn hóa: đảm bảo mọi visit là list[int] không rỗng ===
            cleaned_visits = []
            for visit in visits:
                if not visit:
                    visit = [self.n_codes]
                # Đảm bảo visit là list[int], không chứa list lồng
                clean_visit = []
                for code in visit:
                    if isinstance(code, (list, tuple, np.ndarray)):
                        clean_visit.extend([int(c) for c in code if isinstance(c, (int, np.integer))])
                    elif isinstance(code, (int, np.integer)):
                        clean_visit.append(int(code))
                if not clean_visit:
                    clean_visit = [self.n_codes]
                cleaned_visits.append(clean_visit)
    
            # === Tìm max length và padding ===
            max_len = max(len(v) for v in cleaned_visits)
            padded = []
            for v in cleaned_visits:
                if len(v) < max_len:
                    v = v + [self.n_codes] * (max_len - len(v))
                padded.append(torch.LongTensor(v).to(device))
    
            visit_tensor = torch.stack(padded)  # (T, max_len) – giờ 100% cùng shape!
    
            # === Embedding + sum ===
            emb = self.emb(visit_tensor)           # (T, max_len, D)
            emb = self.dropout(emb)
            visit_emb = emb.sum(dim=1)             # (T, D)
    
            # === RETAIN attention ===
            g, _ = self.alpha_gru(visit_emb.unsqueeze(0))
            h, _ = self.beta_gru(visit_emb.unsqueeze(0))
    
            alpha = F.softmax(self.alpha_lin(g.squeeze(0)), dim=0)  # (T, 1)
            beta = torch.tanh(self.beta_lin(h.squeeze(0)))         # (T, D)
            context = torch.sum(alpha * beta * visit_emb, dim=0)   # (D,)
    
            all_contexts.append(context)
    
        context_batch = torch.stack(all_contexts)  # (B, D)
        return self.output(context_batch)         # (B, n_codes)

    # def forward(self, visits_batch):
    #     """
    #     visits_batch: List[ List[List[int]] ] – batch bệnh nhân, mỗi bệnh nhân có nhiều visit
    #     """
    #     device = next(self.parameters()).device
    #     batch_size = len(visits_batch)
        
    #     all_contexts = []
    #     for visits in visits_batch:  # Duyệt từng bệnh nhân trong batch
    #         if not visits:
    #             # Nếu không có visits, dùng code cuối cùng
    #             visits = [[self.n_codes - 1]]
                
    #         # Step 1: Tạo visit embedding
    #         visit_embs = []
    #         for visit in visits:
    #             if not visit:
    #                 visit = [self.n_codes - 1]
    #             codes = torch.LongTensor(visit).to(device)
    #             emb = self.dropout(self.emb(codes))        # (n_codes_in_visit, D)
    #             v_emb = emb.sum(dim=0)                     # (D,)
    #             visit_embs.append(v_emb)
            
    #         visit_tensor = torch.stack(visit_embs)             # (T, D)
            
    #         # Step 2: RETAIN attention – CHUẨN GỐC PAPER
    #         g, _ = self.alpha_gru(visit_tensor.unsqueeze(0))   # (1, T, D)
    #         h, _ = self.beta_gru(visit_tensor.unsqueeze(0))
            
    #         alpha = F.softmax(self.alpha_lin(g.squeeze(0)), dim=0)  # (T, 1) ← softmax theo visit
    #         beta  = torch.tanh(self.beta_lin(h.squeeze(0)))         # (T, D)
            
    #         context = torch.sum(alpha * beta * visit_tensor, dim=0)  # (D,)
    #         all_contexts.append(context)
        
    #     # Step 3: Gom toàn batch
    #     context_batch = torch.stack(all_contexts)          # (B, D)
    #     logits = self.output(context_batch)                # (B, n_codes)
    #     return logits.squeeze(0) if batch_size == 1 else logits
