import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class Mamba2_PyTorch(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=64,
        d_conv=4,
        expand=2,
        headdim=64,         # IMPORTANT: set 64 for d_model=256 → 4 heads
        ngroups=None,       # If None → ngroups = nheads
        A_init_range=(1, 16),
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        learnable_init_states=False,
        activation="swish",
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = expand * d_model     # 512
        self.headdim = headdim              # 64
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim  # 512/64 = 8 heads

        # IMPORTANT
        self.ngroups = self.nheads if ngroups is None else ngroups 
        assert self.ngroups == self.nheads, "ngroups must equal nheads"

        # [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.nheads * self.d_state + self.nheads
        self.in_proj = nn.Linear(d_model, d_in_proj)

        # Depthwise Conv1D
        conv_dim = self.d_inner + 2 * self.nheads * self.d_state
        self.conv1d = nn.Conv1d(
            conv_dim, conv_dim, 
            kernel_size=d_conv, 
            groups=conv_dim, 
            padding=d_conv-1
        )

        # Activation
        self.act = nn.SiLU()

        # dt init
        dt = torch.exp(
            torch.rand(self.nheads)*(math.log(dt_max)-math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))

        # A parameter
        A = torch.empty(self.nheads).uniform_(*A_init_range)
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)

        # Skip
        self.D = nn.Parameter(torch.ones(self.nheads))

        # learnable init states
        self.learnable_init_states = learnable_init_states
        if learnable_init_states:
            self.init_states = nn.Parameter(
                torch.zeros(self.nheads, self.headdim, self.d_state)
            )

        # critical normalization
        self.norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model)


    def SSM_update(self, x, B, C, dt, A):
        """
        x: (B,L,nheads,headdim)
        B: (B,L,nheads,d_state)
        C: (B,L,nheads,d_state)
        dt: (B,L,nheads)
        A: (nheads)
        """
        Bx = x.size(0)
        h = torch.zeros(x.size(0), self.nheads, self.d_state, device=x.device)
        A = A.view(1, self.nheads, 1)

        outputs = []
        for t in range(x.size(1)):
            # V = x, B = Bmat, C = Cmat 

            # V_t → (B,nheads,headdim)
            # B_t → (B,nheads,d_state)
            v_t = x[:,t]      # (B,nheads,headdim)
            B_t = B[:,t]      # (B,nheads,d_state)
            C_t = C[:,t]      # (B,nheads,d_state)

            # B_t * v_t mean-over-headdim
            inp = torch.einsum("bhd,bhd->bh", v_t, B_t)  # (B,nheads)
            inp = inp.unsqueeze(-1)

            h = torch.exp(A)*h + inp

            y_t = torch.einsum("bhd,bhd->bh", h, C_t)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # (B,L,nheads)
        return y


    def forward(self, u):
        B, L, D = u.size()

        # project input
        zxbcdt = self.in_proj(u)

        # split
        z, xBC, dt = torch.split(
            zxbcdt,
            [self.d_inner, self.d_inner + 2*self.nheads*self.d_state, self.nheads],
            dim=-1
        )

        dt = F.softplus(dt + self.dt_bias)
        A = -torch.exp(self.A_log)

        # Depthwise conv
        xBC = self.act(self.conv1d(xBC.transpose(1,2)).transpose(1,2))

        # split x B C
        x, Bmat, Cmat = torch.split(
            xBC,
            [self.d_inner, self.nheads*self.d_state, self.nheads*self.d_state],
            dim=-1
        )

        # reshape
        x    = rearrange(x,    "b l (h p) -> b l h p", h=self.nheads)
        Bmat = rearrange(Bmat, "b l (h n) -> b l h n", h=self.nheads)
        Cmat = rearrange(Cmat, "b l (h n) -> b l h n", h=self.nheads)

        # SSM compute y
        y = self.SSM_update(x, Bmat, Cmat, dt, A)

        # reshape to (B,L,d_inner)
        y = rearrange(y, "b l h -> b l (h)")

        # gating: multiply with z, then RMSNorm / LayerNorm
        y = self.norm(y) * torch.sigmoid(z)

        out = self.out_proj(y)
        return out
