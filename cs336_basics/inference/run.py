import torch
from typing import Dict
from cs336_basics.base_modules import softmax
from cs336_basics.config import DATA_DIR
from cs336_basics.inference.cache import KVCache
from cs336_basics.tokenizer import BPETokenizer
from cs336_basics.train_modules import load_checkpoint
from cs336_basics.transformer import Transformer

class InferenceEngine:

    def __init__(self, config: Dict):
        self.config = config
        self.device = config.get("device", None)
        self.dtype = config.get("dtype", None)
        inference_config: Dict = config.get("inference_config", {})
        self.max_length = inference_config.get("max_length", 0)
        self.temperature = inference_config.get("temperature", 0.0)
        self.topp = inference_config.get("top-p", 0.0)
        self.kv_cache = None
        self._setup_model()
        self._setup_tokenizer()

    def _setup_model(self):
        assert "model_config" in self.config.keys()
        model_config = self.config["model_config"]
        assert model_config["name"] == "transformer"
        num_layers = model_config["num_layers"]
        context_length = model_config["context_length"]
        num_heads = model_config["num_heads"]
        d_attn = model_config["d_model"]
        if self.config["enable_kv_cache"]:
            self.kv_cache = KVCache((2, num_layers, 1, num_heads, context_length, d_attn//num_heads),
                                        dtype=self.dtype, device=self.device)
        self.model = Transformer(
            vocab_size=model_config["vocab_size"],
            d_embedding=model_config["d_embedding"],
            num_heads=num_heads,
            d_attn=d_attn,
            d_ff=model_config["d_ff"],
            num_layers=num_layers,
            context_length=context_length,
            theta=model_config["theta"],
            device=self.device,
            dtype=self.dtype
        )
        load_checkpoint(src="output/model.pt", model=self.model)

    def _setup_tokenizer(self):
        assert "tokenizer_config" in self.config.keys()
        tokenizer_config = self.config["tokenizer_config"]
        self.tokenizer = BPETokenizer.from_file(
            vocab_file=tokenizer_config["vocab_file"],
            merges_file=tokenizer_config["merge_file"],
            special_tokens=tokenizer_config["special_tokens"]
        )

    def inference(self, x: str):
        print(x, end='')
        encoded_input = self.tokenizer.encode(x)
        input_tensor = torch.Tensor(encoded_input).unsqueeze(0).to(device=self.device, dtype=torch.int64) # (bs, seq_len)
        output_len = input_tensor.size(1)
        while output_len < self.max_length:
            next = self._decode(input_tensor)
            next_token = self.tokenizer.decode(next.to(dtype=torch.int64).flatten().tolist())
            print(next_token, end='')
            if next_token == '<|endoftext|>':
                break
            if self.kv_cache is not None:
                self.kv_cache.is_prefill = False
                self.kv_cache.cached_seq_len += input_tensor.size(1)
                input_tensor = next
            else:
                input_tensor = torch.cat([input_tensor, next], dim=-1)
            output_len += 1
        print()


    def _decode(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape = (bs, seq_len)
        logit = self.model.forward(x, self.kv_cache)
        output = softmax(logit, dim=-1) # shape = (bs, seq_len, vocab_size)
        prob = output[:,-1,:] # (bs, vocab_size)
        ids = torch.argmax(prob, dim=-1).unsqueeze(1) # shape = (bs, 1)
        return ids


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    model_config = {
        "name": "transformer",
        "vocab_size": 10000,
        "d_embedding": 512,
        "num_heads": 16,
        "d_model": 512,
        "d_ff": 1344,
        "num_layers": 4,
        "context_length": 256,
        "theta": 10000,
    }
    tokenizer_config = {
        "vocab_file": f"{DATA_DIR}/vocab/TinyStoriesV2-GPT4-train_vocab.json",
        "merge_file": f"{DATA_DIR}/vocab/TinyStoriesV2-GPT4-train_merges.txt",
        "special_tokens": ["<|endoftext|>"]
    }
    config = {
        "model_config": model_config,
        "tokenizer_config": tokenizer_config,
        "inference_config": {
            "max_length": 256,
            "temperature": 0.0,
            "top-p": 0.0
        },
        "device": device,
        "dtype": dtype,
        "enable_kv_cache": True,
    }
    inference = InferenceEngine(config)
    prompt = "Once upon a time there was a little boy"
    inference.inference(prompt)