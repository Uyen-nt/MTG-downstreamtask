class MICRON_DX(nn.Module):
    def __init__(self, vocab_size, emb_dim=256, device=torch.device('cpu')):
        super(MICRON_DX, self).__init__()

        self.device = device

        # Embedding cho diagnosis codes
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.dropout = nn.Dropout(p=0.5)

        # health representation
        self.health_net = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
        )

        # dự đoán diagnoses cho visit tiếp theo
        self.predictor = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2),
            nn.ReLU(),
            nn.Linear(emb_dim * 2, vocab_size)
        )

        self.init_weights()

    def forward(self, input):
        """
        input dạng:
        [
            [codes of visit 1],
            [codes of visit 2],
            ...
        ]
        L-1 visits 
        predict visit L
        """

        # visit hiện tại (t)
        diag_emb_t = self.embedding(torch.LongTensor(input[-1]).to(self.device))  
        diag_emb_t = diag_emb_t.mean(dim=0, keepdim=True)   # (1, dim)

        # visit trước đó (t-1), nếu không có → zeros
        if len(input) < 2:
            diag_emb_prev = torch.zeros_like(diag_emb_t)
        else:
            diag_emb_prev = self.embedding(torch.LongTensor(input[-2]).to(self.device))
            diag_emb_prev = diag_emb_prev.mean(dim=0, keepdim=True)

        # transform
        h_t = self.health_net(diag_emb_t)       # (1,dim)
        h_prev = self.health_net(diag_emb_prev) # (1,dim)

        # residual learning
        h_res = h_t - h_prev

        # predict diagnoses
        logits = self.predictor(h_res)  # (1, vocab)

        return logits

    def init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        for m in self.predictor:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
