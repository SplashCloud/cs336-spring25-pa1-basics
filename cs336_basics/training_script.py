from typing import Dict, Any
import torch
from torch import argmax
from tqdm import tqdm
import numpy as np
import os
import time
from dotenv import load_dotenv
from concurrent.futures import ProcessPoolExecutor

from cs336_basics.logger import LoggerManager
from cs336_basics.transformer import Transformer
from cs336_basics.optimizer import AdamW, SGD
from cs336_basics.tokenizer import BPETokenizer
from cs336_basics.train_modules import cross_entropy, data_loading
from cs336_basics.base_modules import softmax


load_dotenv()
dir = os.getenv('DATA_DIR')

'''
1. model
2. optimizer
3. tokenizer: encode input + decode output
4. vocabulary from tokenizer
5. dataloader
'''

# Global function for multiprocessing (must be at module level)
def _encode_chunk_worker(args):
    """Worker function for multiprocessing encoding"""
    lines_chunk, tokenizer_config = args
    # Create tokenizer in each worker process
    tokenizer = BPETokenizer.from_file(
        vocab_file=tokenizer_config["vocab_file"],
        merges_file=tokenizer_config["merge_file"],
        special_tokens=tokenizer_config["special_tokens"]
    )
    result = []
    for line in lines_chunk:
        encoded = tokenizer.encode(line)
        result.extend(encoded)
    return result

class Training:

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = LoggerManager(name="TrainingLogger", log_to_file=True, log_file_path="logs/train-try-1.log")
        self._setup_model()
        self._setup_optimizer()
        self._setup_tokenizer()
        self._setup_dataloader()
        self.logger.info("Training Object has done initialization...")

    def _setup_model(self):
        assert "model_config" in self.config.keys()
        model_config = self.config["model_config"]
        assert model_config["name"] == "transformer"
        self.model = Transformer(
            vocab_size=model_config["vocab_size"],
            d_embedding=model_config["d_embedding"],
            num_heads=model_config["num_heads"],
            d_attn=model_config["d_model"],
            d_ff=model_config["d_ff"],
            num_layers=model_config["num_layers"],
            context_length=model_config["context_length"],
            theta=model_config["theta"],
            device=model_config["device"],
            dtype=model_config["dtype"]
        )

    def _setup_optimizer(self):
        assert "optimizer_config" in self.config.keys()
        opti_config = self.config["optimizer_config"]
        if opti_config["name"] == "adamw":
            self.opti = AdamW(
                params=self.model.parameters(),
                lr=opti_config["lr"],
                weight_decay=opti_config["weight_decay"],
                betas=(opti_config["beta1"], opti_config["beta2"],),
                eps=opti_config["eps"]
            )
        else:
            self.opti = SGD(
                parameters=self.model.parameters(),
                lr=opti_config["lr"]
            )

    def _setup_tokenizer(self):
        assert "tokenizer_config" in self.config.keys()
        tokenizer_config = self.config["tokenizer_config"]
        self.tokenizer = BPETokenizer.from_file(
            vocab_file=tokenizer_config["vocab_file"],
            merges_file=tokenizer_config["merge_file"],
            special_tokens=tokenizer_config["special_tokens"]
        )

    def _setup_dataloader(self):
        assert "dataloader_config" in self.config.keys()
        dataloader_config = self.config["dataloader_config"]
        encoded_input = []
        txt_file = dataloader_config["input_file"] # xxx.txt
        npy_file = txt_file.split(".")[0] + ".npy"
        if os.path.exists(npy_file):
            encoded_input = np.load(npy_file)
        else:
            s = time.time()
            # Read all lines first (this is fast)
            self.logger.info("Reading file lines...")
            with open(dataloader_config["input_file"], "r") as f:
                lines = f.readlines()
            
            # Use multi-processing to encode (bypasses GIL for CPU-intensive tasks)
            max_available_processes = os.cpu_count() or 1  # Get available CPU cores
            num_processes = min(max_available_processes, len(lines) // 1000 + 1)  # Adjust based on data size
            chunk_size = max(1, len(lines) // num_processes)
            
            self.logger.info(f"Encoding with {num_processes} processes, chunk size: {chunk_size} lines")
            
            # Split lines into chunks
            line_chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]
            
            # Prepare arguments for worker function (tokenizer config + line chunks)
            tokenizer_config = self.config["tokenizer_config"]
            worker_args = [(chunk, tokenizer_config) for chunk in line_chunks]
            
            # Process chunks in parallel using multiprocessing (maintain order)
            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                # Use map to maintain order
                chunk_results = list(tqdm(
                    executor.map(_encode_chunk_worker, worker_args),
                    total=len(worker_args),
                    desc="Encoding chunks"
                ))
                
                # Flatten results
                for chunk_result in chunk_results:
                    encoded_input.extend(chunk_result)
            
            np.save(npy_file, np.array(encoded_input))
            e = time.time()
            self.logger.info(f'encode the input file cost {e - s} seconds')
        self.logger.info(f"dataset has {len(encoded_input)} tokens")
        def loading():
            return data_loading(
                        x=encoded_input,
                        batch_size=dataloader_config["batch_size"],
                        context_length=dataloader_config["context_length"],
                        device=dataloader_config["device"]
                    )
        self.dataloader = loading

    def train(self):
        epochs = self.config["epochs"]
        iterations = self.config["iterations"]
        self.logger.info(f"begin to train for {epochs} epochs, and each has {iterations} iterations")
        try:
            for i in range(epochs):
                for j in tqdm(range(iterations)):
                    input, target = self.dataloader()
                    output = self.model.forward(input)
                    loss = cross_entropy(
                        output.view(-1, output.size(-1)),
                        target.view(-1)
                    )
                    self.logger.log_loss(i, j, loss.item())
                    self.opti.zero_grad()
                    loss.backward()
                    self.opti.step()
        except Exception as e:
            self.logger.error(f'meet exception {e} in the training...')

    def decode(self, x: torch.Tensor):
        # x.shape = (bs, seq_len)
        logit = self.model.forward(x)
        output = softmax(logit, dim=-1) # shape = (bs, seq_len, vocab_size)
        prob = output[:,-1,:]
        ids = argmax(prob, dim=-1) # shape = (bs,)
        # concat the x and ids
        ids = ids.unsqueeze(1)
        seq = torch.cat([x, ids], dim=-1)
        results = []
        for s in seq:
            results.append(self.tokenizer.decode(s.to(dtype=torch.int64).flatten().tolist()))
        return results


if __name__ == "__main__":
    device = torch.device("cuda:0")
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
        "device": device,
        "dtype": dtype
    }
    optimizer_config = {
        "name": "adamw",
        "lr": 1e-3,
        "weight_decay": 0.01,
        "beta1": 0.99,
        "beta2": 0.99,
        "eps": 1e-6
    }
    tokenizer_config = {
        "vocab_file": f"{dir}/data/vocab/TinyStoriesV2-GPT4-train_vocab.json",
        "merge_file": f"{dir}/data/vocab/TinyStoriesV2-GPT4-train_merges.txt",
        "special_tokens": ["<|endoftext|>"]
    }
    dataloader_config = {
        "batch_size": 32,
        "context_length": 256,
        "input_file": f"{dir}/data/TinyStoriesV2-GPT4-train.txt",
        "device": device
    }
    config = {
        "model_config": model_config,
        "optimizer_config": optimizer_config,
        "tokenizer_config": tokenizer_config,
        "dataloader_config": dataloader_config,
        "epochs": 1,
        "iterations": 40000,
    }
    training = Training(config)
    training.train()
