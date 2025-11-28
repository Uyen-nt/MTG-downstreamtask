# gpt/config.py
class GPTConfig:
    def __init__(
        self,
        total_vocab_size,      # sẽ được override sau khi biết vocab thực
        n_positions=1024,
        n_ctx=1024,
        n_embd=768,
        n_layer=8, # giảm 12 xuống 8
        n_head=12,
        layer_norm_epsilon=1e-5,
        batch_size=12,              # giảm 16 xuống 12
        epoch=10, # giảm 100 xuống 10 để test
        lr=3e-4,
        warmup_steps=1000,
        pad_token_id=9999,
        eos_token_id=9998,          # End of Visit
    ):
        self.total_vocab_size = total_vocab_size
        self.n_positions = n_positions
        self.n_ctx = n_ctx
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.layer_norm_epsilon = layer_norm_epsilon
        self.batch_size = batch_size
        self.epoch = epoch
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.pad_token_id = total_vocab_size - 1
        self.eos_token_id = total_vocab_size - 2
