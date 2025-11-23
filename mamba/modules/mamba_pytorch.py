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
        learnable_init_states=False,   # thêm vào
        activation="swish",
        chunk_size=None,               # thêm vào
        use_mem_eff_path=None,         # thêm vào
        layer_idx=None,                # thêm vào
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = expand * d_model
        self.headdim = headdim
        self.ngroups = ngroups
        self.nheads = self.d_inner // self.headdim

        d_in_proj = 2 * self.d_inner + 2 * ngroups * d_state + self.nheads
        self.in_proj = nn.Linear(d_model, d_in_proj)

        conv_dim = self.d_inner + 2 * ngroups * d_state
        self.conv1d = nn.Conv1d(conv_dim, conv_dim, kernel_size=d_conv, groups=conv_dim)

        self.act = nn.SiLU()

        dt = torch.exp(
            torch.rand(self.nheads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))

        self.A_log = nn.Parameter(torch.log(torch.empty(self.nheads).uniform_(*A_init_range)))
        self.D = nn.Parameter(torch.ones(self.nheads))

        self.norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model)

        self.learnable_init_states = learnable_init_states
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx
        print(">>>> Mamba2_PyTorch initialized with:",
      "learnable_init_states=", learnable_init_states,
      "chunk_size=", chunk_size,
      "use_mem_eff_path=", use_mem_eff_path,
      "layer_idx=", layer_idx)




    def SSM_update(self, x, B, C, dt, A):

        # reshape EXACT LIKE ORIGINAL
        x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)     # (B, L, nheads, headdim)
        B = rearrange(B, "b l (h n) -> b l h n", h=self.nheads)       # (B, L, nheads, d_state)
        C = rearrange(C, "b l (h n) -> b l h n", h=self.nheads)       # (B, L, nheads, d_state)
    
        # hidden state
        h = torch.zeros(x.size(0), self.nheads, self.d_state, device=x.device)
    
        A = A.view(1, self.nheads, 1)    # (1, nheads, 1)
    
        outputs = []
    
        for t in range(x.size(1)):
            # B*x for each head over headdim
            inp = torch.einsum('bhn,bhp->bh', B[:,t], x[:,t])  # (B, nheads)
            inp = inp.unsqueeze(-1)                             # (B, nheads, 1)
    
            h = torch.exp(A) * h + inp
    
            y_t = torch.einsum('bhn,bhn->bh', C[:,t], h)        # (B, nheads)
            outputs.append(y_t)
    
        y = torch.stack(outputs, dim=1)      # (batch, seq, nheads)
    
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

        xBC = self.act(self.conv1d(xBC.transpose(1,2)).transpose(1,2))
        x, Bmat, Cmat = torch.split(
            xBC, [self.d_inner, self.ngroups*self.d_state, self.ngroups*self.d_state], dim=-1
        )

        y = self.SSM_update(x, Bmat, Cmat, dt, A)

        y = self.norm(y) * torch.sigmoid(z)
        return self.out_proj(y)
