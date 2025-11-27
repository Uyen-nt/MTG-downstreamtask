# retain_micron/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RETAIN_Diagnosis(nn.Module):
    def __init__(self, n_codes, emb_size=256, dropout=0.3):
        super().__init__()
        self.n_codes = n_codes
        self.padding_idx = n_codes

        self.emb = nn.Embedding(n_codes + 1, emb_size, padding_idx=self.padding_idx)
        #self.visit_norm = nn.LayerNorm(emb_size)
        self.dropout = nn.Dropout(dropout)

        self.alpha_gru = nn.GRU(emb_size, emb_size, batch_first=True)
        self.beta_gru = nn.GRU(emb_size, emb_size, batch_first=True)

        self.alpha_lin = nn.Linear(emb_size, 1)
        self.beta_lin = nn.Linear(emb_size, emb_size)
        self.output = nn.Linear(emb_size, n_codes)
        
        # Better initialization
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lin' in name or 'output' in name:
                    nn.init.xavier_uniform_(param)
                elif 'emb' in name:
                    nn.init.normal_(param, mean=0, std=0.1)
            elif 'bias' in name:
                if 'output' in name:
                    nn.init.constant_(param, 0.0)  
                else:
                    nn.init.constant_(param, 0.0)

    def forward(self, visits_batch):
        device = next(self.parameters()).device
        all_contexts = []
        
        for visits in visits_batch:           
            if not visits:
                visits = [[self.padding_idx]]

            visit_embs = []
            for visit in visits:
                clean_visit = self._flatten_visit(visit)
                codes = torch.LongTensor(clean_visit).to(device)
                emb = self.emb(codes)
                emb = self.dropout(emb)
                #emb = self.visit_norm(emb)
                
                v_emb = emb.mean(dim=0)             
                visit_embs.append(v_emb)

            if not visit_embs:
                visit_embs = [self.emb(torch.LongTensor([self.padding_idx]).to(device)).squeeze()]
                
            visit_tensor = torch.stack(visit_embs)  # (T, D)

            # RETAIN attention
            g, _ = self.alpha_gru(visit_tensor.unsqueeze(0))  # (1, T, D)
            h, _ = self.beta_gru(visit_tensor.unsqueeze(0))   # (1, T, D)

            alpha = F.softmax(self.alpha_lin(g.squeeze(0)), dim=0)  # (T, 1)
            beta = torch.tanh(self.beta_lin(h.squeeze(0)))          # (T, D)
            context = torch.sum(alpha * beta * visit_tensor, dim=0)  # (D,)
            all_contexts.append(context)

        context_batch = torch.stack(all_contexts)  # (B, D)
        output = self.output(context_batch)        # (B, n_codes)
        
        return output

    def _flatten_visit(self, visit):
        """Sửa lỗi xử lý list lồng nhau"""
        if not visit:
            return [self.padding_idx]
        
        # DEBUG: Kiểm tra cấu trúc visit
        if isinstance(visit[0], (list, np.ndarray)):
            # Flatten nested list
            flat = []
            for sublist in visit:
                if isinstance(sublist, (list, np.ndarray)):
                    flat.extend([int(x) for x in sublist if isinstance(x, (int, np.integer))])
                elif isinstance(sublist, (int, np.integer)):
                    flat.append(int(sublist))
            return flat if flat else [self.padding_idx]
        else:
            # Đã là list trực tiếp
            flat = [int(x) for x in visit if isinstance(x, (int, np.integer))]
            return flat if flat else [self.padding_idx]



# class RETAIN_Diagnosis(nn.Module):
#     def __init__(self, n_codes, emb_size=256, dropout=0.5):
#         super().__init__()
#         self.n_codes = n_codes
#         self.padding_idx = n_codes

#         self.emb = nn.Embedding(n_codes + 1, emb_size, padding_idx=self.padding_idx)
#         self.dropout = nn.Dropout(dropout)

#         self.alpha_gru = nn.GRU(emb_size, emb_size, batch_first=True)
#         self.beta_gru  = nn.GRU(emb_size, emb_size, batch_first=True)

#         self.alpha_lin = nn.Linear(emb_size, 1)
#         self.beta_lin  = nn.Linear(emb_size, emb_size)
#         self.output    = nn.Linear(emb_size, n_codes)
#         self._init_weights()

#     def _init_weights(self):
#         for name, param in self.named_parameters():
#             if 'weight' in name:
#                 if 'lin' in name or 'output' in name:
#                     nn.init.xavier_uniform_(param)
#                 elif 'emb' in name:
#                     nn.init.normal_(param, mean=0, std=0.1)
#             elif 'bias' in name:
#                 nn.init.constant_(param, 0.1)


#     def forward(self, visits_batch):
#         device = next(self.parameters()).device
#         batch_size = len(visits_batch)
#         all_contexts = []
        
#         for i, visits in enumerate(visits_batch):           
#             if not visits:
#                 visits = [[self.padding_idx]]

#             visit_embs = []
#             for visit in visits:
#                 clean_visit = self._flatten_visit(visit)
#                 codes = torch.LongTensor(clean_visit).to(device)
#                 emb = self.emb(codes)
#                 emb = self.dropout(emb)
#                 v_emb = emb.sum(dim=0)
#                 visit_embs.append(v_emb)

#             if not visit_embs:
#                 visit_embs = [self.emb(torch.LongTensor([self.padding_idx]).to(device)).squeeze()]
                
#             visit_tensor = torch.stack(visit_embs)  # (T, D)

#             # RETAIN attention với batch dimension đúng
#             g, _ = self.alpha_gru(visit_tensor.unsqueeze(0))  # (1, T, D)
#             h, _ = self.beta_gru(visit_tensor.unsqueeze(0))   # (1, T, D)

#             alpha = F.softmax(self.alpha_lin(g.squeeze(0)), dim=0)  # (T, 1)
#             beta = torch.tanh(self.beta_lin(h.squeeze(0)))          # (T, D)
#             context = torch.sum(alpha * beta * visit_tensor, dim=0)  # (D,)
#             all_contexts.append(context)

#         context_batch = torch.stack(all_contexts)  # (B, D)
#         output = self.output(context_batch)        # (B, n_codes)
        
#         return output

#     def _flatten_visit(self, visit):
#         """Sửa lỗi xử lý list lồng nhau"""
#         if not visit:
#             return [self.padding_idx]
        
#         # DEBUG: Kiểm tra cấu trúc visit
#         if isinstance(visit[0], (list, np.ndarray)):
#             # print(f"WARNING: Nested list detected in visit! Flattening...")
#             # print(f"Original: {visit[:2]}...")
#             # Flatten nested list
#             flat = []
#             for sublist in visit:
#                 if isinstance(sublist, (list, np.ndarray)):
#                     flat.extend([int(x) for x in sublist if isinstance(x, (int, np.integer))])
#                 elif isinstance(sublist, (int, np.integer)):
#                     flat.append(int(sublist))
#             # print(f"Flattened: {flat[:5]}...")
#             return flat if flat else [self.padding_idx]
#         else:
#             # Đã là list trực tiếp
#             flat = [int(x) for x in visit if isinstance(x, (int, np.integer))]
#             return flat if flat else [self.padding_idx]

