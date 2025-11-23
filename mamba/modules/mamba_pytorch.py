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
        conv_init=None,
        expand=2,
        headdim=128,
        ngroups=1,
        A_init_range=(1, 16),
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        learnable_init_states=False,
        activation="swish",
        bias=False,
        conv_bias=True,
        chunk_size=256,
        use_mem_eff_path=False,
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = expand * d_model   # = 512
        self.headdim = headdim            # = 128
        self.ngroups = ngroups            # = 1
        self.nheads = self.d_inner // self.headdim  # = 4

        # [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * ngroups * d_state + self.nheads
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=bias)

        conv_dim = self.d_inner + 2 * ngroups * d_state
        self.conv1d = nn.Conv1d(conv_dim, conv_dim, kernel_size=d_conv,
                                groups=conv_dim, bias=conv_bias, padding=d_conv - 1)

        if conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -conv_init, conv_init)

        self.learnable_init_states = learnable_init_states
        if learnable_init_states:
            self.init_states = nn.Parameter(torch.zeros(self.nheads, self.headdim, self.d_state))

        self.act = nn.SiLU()

        # dt init
        dt = torch.exp(
            torch.rand(self.nheads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))

        # A parameter
        A = torch.empty(self.nheads).uniform_(*A_init_range)
        self.A_log = nn.Parameter(torch.log(A))

        # skip param
        self.D = nn.Parameter(torch.ones(self.nheads))

        self.norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model)


    def SSM_update(self, x, B, C, dt, A):
        """
        x: (B, L, d_inner) -> reshape → (B, L, nheads, headdim)
        B: (B, L, nheads * d_state)
        C: (B, L, nheads * d_state)
        """

        B = rearrange(B, "b l (h n) -> b l h n", h=self.nheads)
        C = rearrange(C, "b l (h n) -> b l h n", h=self.nheads)
        x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)

        # initial state
        h = torch.zeros(x.size(0), self.nheads, self.d_state, device=x.device)

        A = A.unsqueeze(0).unsqueeze(-1)     # (1, nheads, 1)
        outputs = []

        for t in range(x.size(1)):
            # B * x  over headdim:
            # B: (B, nheads, d_state)
            # x: (B, nheads, headdim)
            # WRONG: (B,nheads,1)  -> anh từng làm thế!!!
            # CORRECT: (B,nheads,d_state) each channel learns separately

            inp = torch.matmul(x[:,t], B[:,t].transpose(2,1)) # (B, nheads, nheads)
            inp = inp[:,torch.arange(self.nheads),torch.arange(self.nheads)] 
            inp = inp.unsqueeze(-1)

            h = torch.exp(A) * h + inp

            # y_t = C*h
            y_t = torch.sum(C[:,t] * h, dim=-1)   # (B, nheads)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # (B, L, nheads)
        y = repeat(y, 'b l h -> b l (h p)', p=self.headdim)
        return y 


    def forward(self, u):
        B, L, D = u.size()

        zxbcdt = self.in_proj(u)

        z, xBC, dt = torch.split(
            zxbcdt, [self.d_inner,
                     self.d_inner + 2*self.ngroups*self.d_state,
                     self.nheads], dim=-1
        )
        dt = F.softplus(dt + self.dt_bias)
        A = -torch.exp(self.A_log)

        xBC = self.act(self.conv1d(xBC.transpose(1,2)).transpose(1,2))

        x, Bmat, Cmat = torch.split(
            xBC, [self.d_inner,
                  self.ngroups*self.d_state*self.nheads,
                  self.ngroups*self.d_state*self.nheads], dim=-1
        )

        y = self.SSM_update(x, Bmat, Cmat, dt, A)

        y = self.norm(y) * torch.sigmoid(z)
        return self.out_proj(y)
