import torch
import torch.nn as nn

class MICRON_DX(nn.Module):
    def __init__(self, vocab_size, emb_dim=256, device=torch.device('cpu')):
        super(MICRON_DX, self).__init__()

        self.device = device

        # Embedding cho diagnosis codes
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.dropout = nn.Dropout(p=0.3)

        self.health_net = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
        )

        self.predictor = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2),
            nn.ReLU(),
            nn.Linear(emb_dim * 2, vocab_size)
        )

        self.init_weights()

    def forward(self, visits):
        """
        visits: list[ list[int] ]
        history: visits[:-1]
        label: visits[-1]
        """

        # visit hiện tại (t)
        diag_emb_t = self.embedding(torch.LongTensor(visits[-1]).to(self.device))  
        diag_emb_t = diag_emb_t.mean(dim=0, keepdim=True)   

        # visit trước đó (t-1)
        if len(visits) < 2:
            diag_emb_prev = torch.zeros_like(diag_emb_t)
        else:
            diag_emb_prev = self.embedding(torch.LongTensor(visits[-2]).to(self.device))
            diag_emb_prev = diag_emb_prev.mean(dim=0, keepdim=True)

        h_t = self.health_net(diag_emb_t)       
        h_prev = self.health_net(diag_emb_prev) 

        # residual learning
        h_res = h_t - h_prev

        logits = self.predictor(h_res)  
        return logits

    def init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        for m in self.predictor:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
