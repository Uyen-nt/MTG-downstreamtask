# retain/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RETAIN_Single(nn.Module):
    def __init__(self, n_codes, emb_size=128, device=None):
        super().__init__()
        self.n_codes = n_codes
        self.emb_size = emb_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ==== embedding cho từng ICD code ====
        self.embedding = nn.Embedding(n_codes + 1, emb_size, padding_idx=n_codes)

        # ==== attention GRUs ====
        self.alpha_gru = nn.GRU(emb_size, emb_size, batch_first=True)
        self.beta_gru  = nn.GRU(emb_size, emb_size, batch_first=True)

        # ==== attention weights ====
        self.alpha_li = nn.Linear(emb_size, 1)
        self.beta_li  = nn.Linear(emb_size, emb_size)

        # ==== output ====
        self.output = nn.Linear(emb_size, n_codes)

    def forward(self, visits_batch):
        """
        visits_batch: List[B] 
        each element is:
            visit list length L
            each visit: list of integer ICD codes
        """
        device = self.device
        all_contexts = []

        for visits in visits_batch:
            T = len(visits)

            # ------------------------------
            # Visit embedding (sum like RETAIN gốc)
            # ------------------------------
            visit_vectors = []
            for visit in visits:
                codes = torch.LongTensor(visit).to(device)
                emb = self.embedding(codes)
                visit_vec = torch.sum(emb, dim=0)   # (D,)
                visit_vectors.append(visit_vec)

            visit_tensor = torch.stack(visit_vectors)  # (T, D)

            # ------------------------------
            # GRU α và β
            # ------------------------------
            g, _ = self.alpha_gru(visit_tensor.unsqueeze(0)) # (1, T, D)
            h, _ = self.beta_gru(visit_tensor.unsqueeze(0))

            g = g.squeeze(0)   # (T, D)
            h = h.squeeze(0)   # (T, D)

            # ------------------------------
            # attention
            # ------------------------------
            alpha = F.softmax(self.alpha_li(g), dim=0)  # (T, 1)
            beta  = torch.tanh(self.beta_li(h))         # (T, D)

            # ------------------------------
            # RETAIN context
            # ------------------------------
            c = torch.sum(alpha * beta * visit_tensor, dim=0)  # (D,)
            all_contexts.append(c)

        context_batch = torch.stack(all_contexts)      # (B, D)
        logits = self.output(context_batch)            # (B, n_codes)
        return logits
