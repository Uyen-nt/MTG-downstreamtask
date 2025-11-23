# mamba_icd_model.py
import torch
import torch.nn as nn
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MixerModel
from mamba_ssm.modules.mamba2_simple import Mamba2Simple
from mamba_ssm.modules.block import Block

class MambaICD(nn.Module):
    def __init__(
        self,
        input_dim=256,          # sau khi bạn embed visit (ví dụ: 76 vitals + 12 labs + demographics → ~200-300)
        d_model=768,
        n_layer=12,
        d_state=64,
        expand=2,
        headdim=64,
        ngroups=8,
        num_codes=4918,         # số ICD-9 khác nhau trong MIMIC-III (thường lấy top ~5000)
        dropout=0.1,
        use_time_embedding=True,
    ):
        super().__init__()
        
        # 1. Input embedding
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 2. Time embedding (rất quan trọng cho EHR!)
        self.use_time_embedding = use_time_embedding
        if use_time_embedding:
            self.time_proj = nn.Linear(1, d_model)  # delta days từ visit trước
        
        # 3. Mamba-2 backbone (dùng Mamba2Simple – nhanh nhất 2025)
        self.backbone = MixerModel(
            d_model=d_model,
            n_layer=n_layer,
            d_intermediate=0,           # không dùng MLP → nhẹ hơn, đủ mạnh cho EHR
            vocab_size=1,               # không dùng, để 1 là được
            ssm_cfg=dict(
                layer="Mamba2",
                d_state=d_state,
                expand=expand,
                headdim=headdim,
                ngroups=ngroups,
                d_conv=4,
                chunk_size=256,
            ),
            rms_norm=True,
            residual_in_fp32=True,
            fused_add_norm=True,
        )
        
        # 4. Classification head
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_codes)
        
        self.loss_fn = nn.BCEWithLogitsLoss()
        
    def forward(self, visit_features, time_deltas=None, labels=None):
        """
        visit_features: (B, L, input_dim)
        time_deltas:    (B, L) hoặc (B, L-1) – khoảng cách ngày giữa các visit
        """
        B, L, _ = visit_features.shape
        
        x = self.input_proj(visit_features)  # (B, L, d_model)
        
        # Thêm time embedding (cực kỳ quan trọng cho next-visit prediction)
        if self.use_time_embedding and time_deltas is not None:
            # time_deltas: (B, L) → pad 0 cho visit đầu
            if time_deltas.dim() == 2:
                time_deltas = time_deltas.unsqueeze(-1)  # (B, L, 1)
            time_emb = self.time_proj(time_deltas.float())
            x = x + time_emb
        
        # Mamba-2 forward
        hidden = self.backbone(x)  # (B, L, d_model)
        
        # Pooling: lấy visit cuối cùng (thường tốt nhất cho next-visit prediction)
        pooled = hidden[:, -1, :]  # (B, d_model)
        # hoặc thử mean pooling: hidden.mean(dim=1)
        
        pooled = self.norm(pooled)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)  # (B, num_codes)
        
        if labels is not None:
            loss = self.loss_fn(logits, labels.float())
            return logits, loss
        return logits
