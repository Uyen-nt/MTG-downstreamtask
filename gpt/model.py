
import copy
import math
import torch
import torch.nn as nn

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root)."""
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        super(Conv1D, self).__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x

# === THAY TOÀN BỘ CLASS Attention BẰNG CÁI NÀY ===
class Attention(nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False):
        super().__init__()
        n_state = nx
        assert n_state % config.n_head == 0

        self.n_head = config.n_head
        self.scale = scale
        self.c_attn = Conv1D(n_state * 3, nx)   # q, k, v
        self.c_proj = Conv1D(n_state, nx)

        # Không tạo bias tĩnh 1024x1024 nữa → tiết kiệm >4GB VRAM
        #self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx), persistent=False)
        # Có thể bỏ luôn dòng trên nếu dùng is_causal=True

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # Q, K, V
        qkv = self.c_attn(x)                              # (B, T, 3*C)
        q, k, v = qkv.split(self.c_attn.nf // 3, dim=2)   # mỗi cái (B, T, C)

        # Multi-head
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, H, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Nếu có past (autoregressive generation)
        if layer_past is not None:
            past_k, past_v = layer_past
            k = torch.cat((past_k, k), dim=-2)
            v = torch.cat((past_v, v), dim=-2)

        present = torch.stack((k, v))

        # === DÙNG FLASH ATTENTION / SDPA → SIÊU NHANH + SIÊU TIẾT KIỆM VRAM ===
        if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            # Cách hiện đại nhất (PyTorch 2.0+)
            att = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True          # ← tự động tạo causal mask, không cần bias tĩnh!
            )
        else:
            # Fallback cũ (vẫn ổn nhưng chậm hơn)
            w = torch.matmul(q, k.transpose(-2, -1))
            if self.scale:
                w = w / math.sqrt(v.size(-1))
            w = w.masked_fill(self.bias[:, :, :T, :k.size(-2)] == 0, float('-inf'))
            w = torch.softmax(w, dim=-1)
            att = torch.matmul(w, v)

        # Merge heads
        att = att.transpose(1, 2).contiguous().view(B, T, C)
        att = self.c_proj(att)
        return att, present

class MLP(nn.Module):
    def __init__(self, n_state, config):  # in MLP: n_state=3072 (4 * n_embd)
        super(MLP, self).__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = gelu

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return h2

class Block(nn.Module):
    def __init__(self, n_ctx, config, scale=False):
        super(Block, self).__init__()
        nx = config.n_embd
        self.ln_1 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.attn = Attention(nx, n_ctx, config, scale)
        self.ln_2 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)

    def forward(self, x, layer_past=None):
        a, present = self.attn(self.ln_1(x), layer_past=layer_past)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x, present

class GPT2Model(nn.Module):
    def __init__(self, config):
        super(GPT2Model, self).__init__()
        self.n_layer = config.n_layer
        self.n_embd = config.n_embd
        self.n_vocab = config.total_vocab_size

        self.code_embed_mat = nn.Embedding(config.total_vocab_size, config.n_embd)
        self.pos_embed_mat = nn.Embedding(config.n_positions, config.n_embd)
        block = Block(config.n_ctx, config, scale=True)
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def forward(self, input_ids, position_ids=None, past=None):
        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_ids.size(1) + past_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        inputs_embeds = self.code_embed_mat(input_ids)
        position_embeds = self.pos_embed_mat(position_ids)
        hidden_states = inputs_embeds + position_embeds
        presents = []
        for block, layer_past in zip(self.h, past):
            hidden_states, present = block(hidden_states, layer_past)
            presents.append(present)
        hidden_states = self.ln_f(hidden_states)
        return hidden_states, presents

class GPTHead(nn.Module):
    def __init__(self, model_embeddings_weights, config):
        super(GPTHead, self).__init__()
        self.n_embd = config.n_embd
        self.set_embeddings_weights(model_embeddings_weights)

    def set_embeddings_weights(self, model_embeddings_weights):
        embed_shape = model_embeddings_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = nn.Parameter(model_embeddings_weights)  # Tied weights

    def forward(self, hidden_state):
        code_logits = self.decoder(hidden_state)
        return code_logits

class GPTModel(nn.Module):
    def __init__(self, config):
        super(GPTModel, self).__init__()
        self.transformer = GPT2Model(config)
        self.ehr_head = GPTHead(self.transformer.code_embed_mat.weight, config)
        self.config = config

    def set_tied(self):
        """Make sure we are sharing the embeddings"""
        self.ehr_head.set_embeddings_weights(self.transformer.code_embed_mat.weight)

    def forward(self, input_ids, position_ids=None, ehr_labels=None, past=None):
        hidden_states, presents = self.transformer(input_ids, position_ids, past)
        code_logits = self.ehr_head(hidden_states)
        if ehr_labels is not None:    
            code_logits = code_logits[:, :-1, :].contiguous()
            ehr_labels = ehr_labels[:, 1:].contiguous()
            ce = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
            loss = ce(code_logits.view(-1, code_logits.size(-1)), ehr_labels.view(-1))
            return loss, code_logits, ehr_labels
        
        return code_logits, presents
