# retain_micron/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RETAIN_Diagnosis(nn.Module):
    def __init__(self, n_codes, emb_size=256, dropout=0.5):
        super().__init__()
        self.n_codes = n_codes
        self.padding_idx = n_codes

        self.emb = nn.Embedding(n_codes + 1, emb_size, padding_idx=self.padding_idx)
        self.dropout = nn.Dropout(dropout)

        self.alpha_gru = nn.GRU(emb_size, emb_size, batch_first=True)
        self.beta_gru  = nn.GRU(emb_size, emb_size, batch_first=True)

        self.alpha_lin = nn.Linear(emb_size, 1)
        self.beta_lin  = nn.Linear(emb_size, emb_size)
        self.output    = nn.Linear(emb_size, n_codes)

    def _flatten_visit(self, visit):
        """Đảm bảo visit là list[int] phẳng, không lồng, không array"""
        if not visit:
            return [self.padding_idx]
        flat = []
        for item in visit:
            if isinstance(item, (list, tuple, np.ndarray)):
                flat.extend([int(x) for x in item if isinstance(x, (int, np.integer, float))])
            elif isinstance(item, (int, np.integer, float)):
                flat.append(int(item))
        return flat if flat else [self.padding_idx]

    def forward(self, visits_batch):
        device = next(self.parameters()).device
        all_contexts = []
        
        print(f"\n=== MODEL DEBUG ===")
        print(f"Batch size: {len(visits_batch)}")
        for i, visits in enumerate(visits_batch[:2]):  # Chỉ debug 2 samples đầu
            print(f"Sample {i}: {len(visits)} visits")
            for j, visit in enumerate(visits):
                print(f"  Visit {j}: {len(visit)} codes - First 3: {visit[:3]}")

        for visits in visits_batch:
            if not visits:
                visits = [[self.padding_idx]]

            visit_embs = []
            for visit in visits:
                # FIX CHÍNH TẠI ĐÂY: làm phẳng + chuyển int
                clean_visit = self._flatten_visit(visit)
                codes = torch.LongTensor(clean_visit).to(device)  # giờ 100% là 1D tensor int
                emb = self.emb(codes)
                emb = self.dropout(emb)
                v_emb = emb.sum(dim=0)  # (D,)
                visit_embs.append(v_emb)

            visit_tensor = torch.stack(visit_embs)  # (T, D) – an toàn tuyệt đối
            # DEBUG: Kiểm tra embeddings
            print(f"Visit tensor shape: {visit_tensor.shape}")
            print(f"Visit tensor norm: {visit_tensor.norm():.4f}")

            # RETAIN attention
            g, _ = self.alpha_gru(visit_tensor.unsqueeze(0))
            h, _ = self.beta_gru(visit_tensor.unsqueeze(0))

            alpha = F.softmax(self.alpha_lin(g.squeeze(0)), dim=0)  # (T, 1)
            beta = torch.tanh(self.beta_lin(h.squeeze(0)))          # (T, D)
            context = torch.sum(alpha * beta * visit_tensor, dim=0)
            all_contexts.append(context)

        context_batch = torch.stack(all_contexts)
        output = self.output(context_batch)
    
        # DEBUG: Kiểm tra output
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
        print(f"Output mean: {output.mean().item():.4f}")
        print("==================\n")
        
        return output



# # retain_micron/model.py
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np

# class RETAIN_Diagnosis(nn.Module):
#     def __init__(self, n_codes, emb_size=256, dropout=0.5):
#         super().__init__()
#         self.n_codes = n_codes
#         self.emb_size = emb_size
        
#         self.emb = nn.Embedding(n_codes + 1, emb_size, padding_idx=n_codes)
#         self.dropout = nn.Dropout(dropout)
        
#         self.alpha_gru = nn.GRU(emb_size, emb_size, batch_first=True)
#         self.beta_gru  = nn.GRU(emb_size, emb_size, batch_first=True)
        
#         self.alpha_lin = nn.Linear(emb_size, 1)
#         self.beta_lin  = nn.Linear(emb_size, emb_size)
#         self.output    = nn.Linear(emb_size, n_codes)


#     def forward(self, visits_batch):
#         """
#         visits_batch: List[ List[List[int]] ] – batch bệnh nhân, mỗi bệnh nhân có nhiều visit
#         """
#         device = next(self.parameters()).device
#         batch_size = len(visits_batch)
        
#         all_contexts = []
#         for visits in visits_batch:  # Duyệt từng bệnh nhân trong batch
#             if not visits:
#                 # Nếu không có visits, dùng code cuối cùng
#                 visits = [[self.n_codes - 1]]
                
#             # Step 1: Tạo visit embedding
#             visit_embs = []
#             for visit in visits:
#                 if not visit:
#                     visit = [self.n_codes - 1]
#                 codes = torch.LongTensor(visit).to(device)
#                 emb = self.dropout(self.emb(codes))        # (n_codes_in_visit, D)
#                 v_emb = emb.sum(dim=0)                     # (D,)
#                 visit_embs.append(v_emb)
            
#             visit_tensor = torch.stack(visit_embs)             # (T, D)
            
#             # Step 2: RETAIN attention – CHUẨN GỐC PAPER
#             g, _ = self.alpha_gru(visit_tensor.unsqueeze(0))   # (1, T, D)
#             h, _ = self.beta_gru(visit_tensor.unsqueeze(0))
            
#             alpha = F.softmax(self.alpha_lin(g.squeeze(0)), dim=0)  # (T, 1) ← softmax theo visit
#             beta  = torch.tanh(self.beta_lin(h.squeeze(0)))         # (T, D)
            
#             context = torch.sum(alpha * beta * visit_tensor, dim=0)  # (D,)
#             all_contexts.append(context)
        
#         # Step 3: Gom toàn batch
#         context_batch = torch.stack(all_contexts)          # (B, D)
#         logits = self.output(context_batch)                # (B, n_codes)
#         return logits.squeeze(0) if batch_size == 1 else logits
