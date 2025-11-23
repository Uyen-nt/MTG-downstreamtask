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
        use_mem_eff_path=False,    # vì không chạy triton
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = self.expand * self.d_model     # EX: 512
        self.headdim = headdim                        # EX: 128
        self.ngroups = ngroups
        self.nheads = self.d_inner // self.headdim    # EX: 512/128=4

        self.learnable_init_states = learnable_init_states
        self.activation = activation
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.dt_limit = dt_limit

        # SAME AS ORIGINAL
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias)

        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            conv_dim,
            conv_dim,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            bias=conv_bias
        )

        self.act = nn.SiLU()

        dt = torch.exp(
            torch.rand(self.nheads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)

        # ORIGINAL
        A = torch.empty(self.nheads).uniform_(*A_init_range)
        self.A_log = nn.Parameter(torch.log(A))

        self.D = nn.Parameter(torch.ones(self.nheads))

        # RMSNorm substitute (since Triton not available)
        self.norm = nn.LayerNorm(self.d_inner)

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)


    def forward(self, u, seq_idx=None):
        B, L, D = u.shape

        # input projection
        zxbcdt = self.in_proj(u)
        A = -torch.exp(self.A_log)

        z, xBC, dt = torch.split(
            zxbcdt,
            [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1,
        )

        dt = F.softplus(dt + self.dt_bias)

        # conv1d
        xBC = self.act(self.conv1d(xBC.transpose(1, 2)).transpose(1, 2))
        xBC = xBC[:, :L, :]

        # ALWAYS 4D — NEVER flatten
        x, Bmat, Cmat = torch.split(
            xBC,
            [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state],
            dim=-1,
        )
        x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)
        Bmat = rearrange(Bmat, "b l (h n) -> b l h n", h=self.nheads)
        Cmat = rearrange(Cmat, "b l (h n) -> b l h n", h=self.nheads)

        # hidden state
        h = torch.zeros(B, self.nheads, self.d_state, device=u.device)
        A = A.view(1, self.nheads, 1)

        outs = []
        for t in range(L):
            inp = torch.einsum('bhn,bhp->bh', Bmat[:,t], x[:,t])
            inp = inp.unsqueeze(-1)
            h = torch.exp(A) * h + inp
            y_t = torch.einsum('bhn,bhn->bh', Cmat[:,t], h)
            outs.append(y_t)

        y = torch.stack(outs, dim=1)
        y = rearrange(y, "b l h -> b l (h)")

        y = self.norm(y) * torch.sigmoid(z)
        out = self.out_proj(y)
        return out
