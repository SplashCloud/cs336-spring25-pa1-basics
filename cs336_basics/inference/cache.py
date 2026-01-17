import torch

class KVCache:

    def __init__(self, shape, dtype = None, device = None):
        # (2, num_layers, bs, num_heads, max_seq_len, d_k)
        self.kv_cache = torch.empty(shape, dtype=dtype, device=device)
        self.k_cache = self.kv_cache[0]
        self.v_cache = self.kv_cache[1]
        self.cached_seq_len = 0
        self.is_prefill = True

    def get_k(self, layer_idx):
        return self.k_cache[layer_idx, :, :, 0:self.cached_seq_len, :]

    def get_v(self, layer_idx):
        return self.v_cache[layer_idx, :, :, 0:self.cached_seq_len, :]

    def set(self, layer_idx, new_k, new_v, seq_len):
        self.k_cache[layer_idx, :, :, self.cached_seq_len:self.cached_seq_len+seq_len, :] = new_k
        self.v_cache[layer_idx, :, :, self.cached_seq_len:self.cached_seq_len+seq_len, :] = new_v