from typing import Dict, Any
import torch
from tqdm import tqdm
import numpy as np
import os
import yaml
import argparse
import re
from concurrent.futures import ProcessPoolExecutor

from cs336_basics.logger import LoggerManager
from cs336_basics.transformer import Transformer
from cs336_basics.optimizer import AdamW, SGD
from cs336_basics.tokenizer import BPETokenizer
from cs336_basics.train_modules import cross_entropy, data_loading, save_checkpoint
from cs336_basics.config import DATA_DIR, TRAINING_LOG_FILE
from cs336_basics.timer import Timer

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


def load_config_from_yaml(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file and process it"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Replace environment variables in string values
    def replace_env_vars(obj):
        if isinstance(obj, dict):
            return {k: replace_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [replace_env_vars(item) for item in obj]
        elif isinstance(obj, str):
            # Replace ${VAR_NAME} with environment variable value
            def replacer(match):
                var_name = match.group(1)
                return os.getenv(var_name, match.group(0))
            return re.sub(r'\$\{(\w+)\}', replacer, obj)
        return obj
    
    config = replace_env_vars(config)
    
    # Convert device string to torch.device
    device_str = config.get("device", "cpu")
    if device_str == "cuda:0" and torch.cuda.is_available():
        device_str = "cuda:0"
    config["device"] = torch.device(device_str)
    
    # Convert dtype string to torch.dtype
    dtype_str = config.get("dtype", "float32")
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    config["dtype"] = dtype_map.get(dtype_str, torch.float32)
    
    # Add device and dtype to model_config
    config["model_config"]["device"] = config["device"]
    config["model_config"]["dtype"] = config["dtype"]
    
    # Add device to dataloader_config
    config["dataloader_config"]["device"] = config["device"]
    
    # Calculate warmup_t and cosine_annealing_t if they are not set
    iterations = config.get("iterations", 10000)
    if "optimizer_config" in config:
        opt_config = config["optimizer_config"]
        if opt_config.get("warmup_t") is None:
            opt_config["warmup_t"] = int(iterations * 0.05)
        if opt_config.get("cosine_annealing_t") is None:
            opt_config["cosine_annealing_t"] = iterations
    
    return config


class Training:

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = LoggerManager(name="TrainingLogger", log_to_file=True, log_file_path=TRAINING_LOG_FILE)
        self._setup_model()
        self._setup_optimizer()
        self._setup_tokenizer()
        self._setup_dataloader()
        self.logger.info("Training Object has done initialization...")

    def _setup_model(self):
        assert "model_config" in self.config.keys()
        model_config = self.config["model_config"]
        assert model_config["name"] == "transformer"
        self.logger.info(f'using {model_config['device']}...')
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
        opti_config: Dict = self.config["optimizer_config"]
        if opti_config["name"] == "adamw":
            self.opti = AdamW(
                params=self.model.parameters(),
                lr=opti_config["min_lr"],
                weight_decay=opti_config["weight_decay"],
                betas=(opti_config["beta1"], opti_config["beta2"],),
                eps=opti_config["eps"],
                enable_lr_schedule=opti_config["enable_lr_schedule"],
                max_lr=opti_config.get("max_lr", None),
                warmup_t=opti_config.get("warmup_t", None),
                cosine_annealing_t=opti_config.get("cosine_annealing_t", None)
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
            with Timer() as t:
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
                    
                    encoded_input = np.array(encoded_input) # error: only integer scalar arrays can be converted to a scalar index

                np.save(npy_file, np.array(encoded_input))
            self.logger.info(f'encode the input file cost {t.elapsed:.3f} seconds')
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
                    with Timer() as t:
                        input, target = self.dataloader()
                    time_dl = t.elapsed
                    
                    with Timer() as t:
                        output = self.model.forward(input) # (bs, seq_len, vocab_size)
                    time_inf = t.elapsed
                    
                    with Timer() as t:
                        loss = cross_entropy(
                            output.view(-1, output.size(-1)),
                            target.view(-1)
                        )
                    time_ce = t.elapsed
                    
                    with Timer() as t:
                        self.logger.log_loss(i, j, loss.item())
                    time_lg = t.elapsed
                    
                    with Timer() as t:
                        self.opti.zero_grad()
                        loss.backward()
                    time_bw = t.elapsed
                    
                    with Timer() as t:
                        self.opti.step()
                    time_opt = t.elapsed
                    
                    self.logger.info(f'dataloder: {time_dl}s, inference: {time_inf}s, loss_comp: {time_ce}s, log: {time_lg}s, backward: {time_bw}s, opt: {time_opt}s in iteration#{j}')
        except Exception as e:
            self.logger.error(f'meet exception {e} in the training...')
        finally:
            save_checkpoint(self.model, self.opti, iterations, "output/model.pt")
            self.logger.info(f'saved the training output...')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a transformer model")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)"
    )
    args = parser.parse_args()
    
    # Load configuration from YAML file
    config = load_config_from_yaml(args.config)
    
    training = Training(config)
    training.train()
