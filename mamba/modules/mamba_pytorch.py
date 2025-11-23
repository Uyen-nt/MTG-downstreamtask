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
        learnable_init_states=False,
        activation="swish",
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = self.expand * self.d_model
        self.headdim = headdim
        self.ngroups = ngroups
        self.nheads = self.d_inner // self.headdim
        
        # Projection: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, d_in_proj)

        # Convolution
        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(conv_dim, conv_dim, kernel_size=d_conv, groups=conv_dim)

        self.act = nn.SiLU()

        # dt init
        dt = torch.exp(
            torch.rand(self.nheads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)

        # A parameter
        A = torch.empty(self.nheads).uniform_(*A_init_range)
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)

        # Skip param
        self.D = nn.Parameter(torch.ones(self.nheads))

        # init hidden state
        self.learnable_init_states = learnable_init_states
        if learnable_init_states:
            self.init_states = nn.Parameter(torch.zeros(self.nheads, self.headdim, self.d_state))

        self.norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model)

    def SSM_update(self, x, B, C, dt, A, init_state=None):
        """
        Pure PyTorch SSM incremental update:
            h[t] = exp(A * dt) * h[t-1] + B * x[t]
            y[t] = C * h[t]
        """

        B = rearrange(B, "b l (g n) -> b l g n", g=self.ngroups)
        C = rearrange(C, "b l (g n) -> b l g n", g=self.ngroups)
        x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)

        B = B.view(x.size(0), x.size(1), self.nheads, self.d_state)
        C = C.view(x.size(0), x.size(1), self.nheads, self.d_state)

        expA = torch.exp(A).view(1, 1, -1)
        expA = expA.to(x.device)

        h = torch.zeros(
            x.size(0),
            self.nheads,
            self.d_state,
            device=x.device,
        ) if init_state is None else init_state.expand(x.size(0), -1, -1)

        outputs = []

        for t in range(x.size(1)):
            h = expA * h + B[:,t] * x[:,t].mean(-1, keepdim=True)
            y_t = (C[:,t] * h).sum(-1)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # (B, L, H)
        y = rearrange(y, "b l h -> b l (h)")  

        return y

    def forward(self, u):
        B, L, D = u.size()

        zxbcdt = self.in_proj(u)

        z, xBC, dt = torch.split(
            zxbcdt, [self.d_inner, self.d_inner + 2*self.ngroups*self.d_state, self.nheads], dim=-1
        )

        dt = F.softplus(dt + self.dt_bias)
        A = -torch.exp(self.A_log)

        xBC = self.act(
            self.conv1d(xBC.transpose(1,2)).transpose(1,2)
        )

        x, Bmat, Cmat = torch.split(
            xBC, [self.d_inner, self.ngroups*self.d_state, self.ngroups*self.d_state], dim=-1
        )

        y = self.SSM_update(x, Bmat, Cmat, dt, A)

        y = self.norm(y) * torch.sigmoid(z)
        return self.out_proj(y)
