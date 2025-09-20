import regex as re
import multiprocessing
import os
from typing import BinaryIO
import json
from tqdm import tqdm
import logging

logger = logging.getLogger(name='train_bpe')
logger.setLevel(level=logging.DEBUG)
file_handler = logging.FileHandler(filename="bpe.log", mode='w')
file_handler.setLevel(level=logging.DEBUG)
logger.addHandler(file_handler)

OPTIMIZATION = True

class BPETrainer:

    def __init__(self):
        # initialize the state
        self.id2bs: dict[int, tuple[bytes]] = {} # key is unchanged
        self.bs2id: dict[tuple[bytes], int] = {} # value is unchanged
        self.unique_bs = 0 # unchanged
        self.bs_freq: dict[int, int] = {} # unchanged
        self.bp_freq: dict[tuple[bytes, bytes], int] = {}
        self.bp_occurence: dict[tuple[bytes, bytes], list[int]] = {}
        # initialize the vocab
        self.vocab_size = 256
        self.vocab: dict[int, bytes] = {}
        for i in range(self.vocab_size):
            self.vocab[i] = bytes([i])

    
    def init_vocab_with_special_tokens(self, special_tokens: list[str]):
        for st in special_tokens:
            self.vocab[self.vocab_size] = st.encode(encoding="utf-8")
            self.vocab_size += 1


    def merge_bytes_seq(self, byte_seq_freq_list: list[dict[tuple[bytes], int]]):
        for byte_seq_freq in byte_seq_freq_list:
            for byte_seq, cnt in byte_seq_freq.items():
                if byte_seq not in self.bs2id.keys():
                    self.bs2id[byte_seq] = self.unique_bs
                    self.id2bs[self.unique_bs] = byte_seq
                    self.unique_bs += 1
                bs_id = self.bs2id[byte_seq]
                if bs_id not in self.bs_freq.keys():
                    self.bs_freq[bs_id] = 0
                self.bs_freq[bs_id] += cnt


    def init_bpe_state(self):
        for bs_id, cnt in tqdm(self.bs_freq.items()):
            bs = self.id2bs[bs_id]
            for i in range(len(bs)-1):
                bp = (bs[i], bs[i+1],)
                if bp not in self.bp_freq.keys():
                    self.bp_freq[bp] = 0
                self.bp_freq[bp] += cnt
                if bp not in self.bp_occurence.keys():
                    self.bp_occurence[bp] = []
                if bs_id not in self.bp_occurence[bp]:
                    self.bp_occurence[bp].append(bs_id)


    def merge_most_freq_bp(self):
        self.max_count = max(list(self.bp_freq.values()))
        self.max_bp = None
        for bp, cnt in self.bp_freq.items():
            if cnt == self.max_count:
                self.max_bp = bp if self.max_bp is None or bp > self.max_bp else self.max_bp
        self.merges.append(self.max_bp)
        self.merged_bytes = self.max_bp[0] + self.max_bp[1]
        self.vocab[self.vocab_size] = self.merged_bytes
        self.vocab_size += 1


    def _is_overlapped(self, bp1, bp2):
        '''
        a, r, a => merge (a, r) => (ar, a)
        r, a, r => merge (a, r) => (r, ar)
        when bp1 = (x, y) and bp2 = (y, x)
        both conditions are satisfied
        '''
        return bp1[0] == bp2[1] or bp1[1] == bp2[0]


    def update_bpe_state(self):
        affected_bs_id_list = self.bp_occurence[self.max_bp]
        del self.bp_occurence[self.max_bp]
        del self.bp_freq[self.max_bp]
        # update the byte_seq
        for bs_id in affected_bs_id_list:
            bs = self.id2bs[bs_id]
            i = 0
            new_bs = ()
            while i < len(bs):
                if bs[i] == self.max_bp[0] and (i+1) < len(bs) and bs[i+1] == self.max_bp[1]:
                    new_bs += (self.merged_bytes,)
                    i += 2
                else:
                    new_bs += (bs[i],)
                    i += 1
            self.id2bs[bs_id] = new_bs
            del self.bs2id[bs]
            self.bs2id[new_bs] = bs_id
        # update the bp_occurence and bp_freq
        # actually we just need to update the byte-pair that is overlapped with the merged-one
        # (a, b) => (x,a) / (b,y) => (x,ab) / (ab,y)
        all_bps = list(self.bp_freq.keys())
        for bp in all_bps:
            if not self._is_overlapped(bp, self.max_bp):
                continue
            old_occurence = self.bp_occurence[bp]
            del self.bp_freq[bp]
            del self.bp_occurence[bp]
            ##### Here is the bug:
            # such as origin text is (a, r, a), merge bp (a, r), so here new_bp should be (ar, a), but will set to (r, ar) by the following code
            # `new_bp = (bp[0], self.merged_bytes,) if bp[1] == self.max_bp[0] else (self.merged_bytes, bp[1],)`
            # Root cause is when merge the bp (x, y), the reverse bp if exists (y, x) satisfy the both conditions of overlapping 
            # (a, r, a, ..., r, a, r, a ...) => (ar, a, ..., r, ar, a, ...) merged-bp=(a,r) bp=(r,a)
            for bs_id0 in old_occurence:
                bs0 = self.id2bs[bs_id0]
                for i in range(len(bs0)-1):
                    if bs0[i] == bp[0] and bs0[i+1] == bp[1]:
                        self._add_bp(bp, bs_id0)
                    elif (bs0[i] == self.merged_bytes and self.max_bp[1] == bp[0]) and bs0[i+1] == bp[1]:
                        new_bp = (self.merged_bytes, bp[1],)
                        self._add_bp(new_bp, bs_id0)
                    elif bs0[i] == bp[0] and (bs0[i+1] == self.merged_bytes and self.max_bp[0] == bp[1]):
                        new_bp = (bp[0], self.merged_bytes,)
                        self._add_bp(new_bp, bs_id0)

    
    def _add_bp(self, new_bp, bs_id):
        if new_bp not in self.bp_freq.keys():
            self.bp_freq[new_bp] = 0
            self.bp_occurence[new_bp] = []
        self.bp_freq[new_bp] += self.bs_freq[bs_id]
        if bs_id not in self.bp_occurence[new_bp]:
            self.bp_occurence[new_bp].append(bs_id)


    def train(self, input_path: str, max_vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        logger.debug("Begin BPE training...")
        self.init_vocab_with_special_tokens(special_tokens)
        escaped_special_tokens = [re.escape(st) for st in special_tokens]
        delimiter = "|".join(escaped_special_tokens)
        with open(input_path, "rb") as f:
            chunks = 100
            chunk_boundaries = self.split_into_chunks(f, chunks, b"<|endoftext|>")
            logger.debug(f"Split the raw data into {len(chunk_boundaries)-1} chunks firstly.")
            args = []
            for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:]):
                args.append((input_path, delimiter, start, end))
            nproc = multiprocessing.cpu_count()
            logger.debug(f"Current machine has {nproc} process cores.")
            with multiprocessing.Pool(processes=nproc) as pool:
                results = pool.starmap(per_tokenization_task, args)
                self.merge_bytes_seq(results)
        logger.debug("Per-tokenization has done...")
        self.merges = []
        self.init_bpe_state()
        logger.debug("Begin merging...")
        with tqdm(initial=self.vocab_size, total=max_vocab_size, desc="Processing") as pbar:
            while self.vocab_size < max_vocab_size:
                self.merge_most_freq_bp()
                self.update_bpe_state()
                pbar.update(1)
        return self.vocab, self.merges
    

    def split_into_chunks(self, file: BinaryIO, chunks: int, special_token: bytes) -> list[int]:
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        chunk_size = file_size // chunks
        chunk_boundaries = [i * chunk_size for i in range(chunks)]
        chunk_boundaries.append(file_size)
        block_size = 4096
        for i in range(1, len(chunk_boundaries)):
            init_bound = chunk_boundaries[i]
            file.seek(init_bound)
            while True:
                content = file.read(block_size)
                if content == b"":
                    chunk_boundaries[i] = file_size
                    break
                found_at = content.find(special_token)
                if found_at != -1:
                    chunk_boundaries[i] = init_bound + found_at
                    break
                init_bound += block_size
        return sorted(set(chunk_boundaries))


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def per_tokenization_task(file: str, delimiter: str, chunk_begin_offset: int, chunk_end_offset: int) -> dict[tuple[bytes], int]:
    with open(file, "rb") as f:
        f.seek(chunk_begin_offset)
        chunk = f.read(chunk_end_offset - chunk_begin_offset).decode(encoding="utf-8", errors="ignore")
        small_chunks = re.split(delimiter, chunk)
        token_freq: dict[str, int] = {}
        for sc in small_chunks:
            iter = re.finditer(PAT, sc)
            for i in iter:
                text = i.group()
                if text not in token_freq.keys():
                    token_freq[text] = 0
                token_freq[text] += 1
        byte_seq_freq: dict[tuple[bytes], int] = {}
        for token, cnt in token_freq.items():
            byte_seq = tuple(bytes([b]) for b in list(token.encode(encoding="utf-8")))
            byte_seq_freq[byte_seq] = cnt
        return byte_seq_freq


def vocab_init(special_tokens: list[str]) -> dict[int, bytes]:
    vocab_size = 256
    vocab: dict[int, bytes] = {}
    for i in range(vocab_size):
        vocab[i] = bytes([i])
    for st in special_tokens:
        vocab[vocab_size] = bytes(st.encode(encoding="utf-8"))
        vocab_size += 1
    return vocab


def per_tokenization(input_path: str, special_tokens: list[str]) -> dict[tuple[bytes], int]:
    escaped_special_tokens = [re.escape(st) for st in special_tokens]
    delimiter = "|".join(escaped_special_tokens)
    byte_seq_freq: dict[tuple[bytes], int] = {}
    with open(input_path, "r") as f:
        content = f.read() # TODO(read by chunk)
        chunks = re.split(delimiter, content)
        nproc = multiprocessing.cpu_count()
        with multiprocessing.Pool(processes=nproc) as pool:
            results = pool.map(per_tokenization_task, chunks)
            for res in results:
                for bs, cnt in res.items():
                    if bs not in byte_seq_freq.keys():
                        byte_seq_freq[bs] = 0
                    byte_seq_freq[bs] += cnt
    return byte_seq_freq


def run(input_path: str,
        max_vocab_size: int,
        special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    # 1. Vocabulary Initialization
    vocab = vocab_init(special_tokens)
    vocab_size = len(vocab)
    assert(max_vocab_size >= vocab_size)
    if vocab_size == max_vocab_size:
        return vocab, []
    
    # 2. Per-Tokenization
    per_token_map = per_tokenization(input_path, special_tokens)

    # 3. Training
    byte_pair_map: dict[tuple[bytes, bytes], int] = {}
    byte_pair_occurence_map: dict[tuple[bytes, bytes], list[tuple[bytes]]] = {}
    merges: list[tuple[bytes, bytes]] = []
    iteration = 0
    while vocab_size < max_vocab_size:
        # 3.1 count all the byte-pairs' frequency
        for byte_seq, cnt in per_token_map.items():
            for i in range(len(byte_seq)-1):
                bp = (byte_seq[i], byte_seq[i+1],)
                if bp not in byte_pair_map.keys():
                    byte_pair_map[bp] = 0
                byte_pair_map[bp] += cnt
                if bp not in byte_pair_occurence_map.keys():
                    byte_pair_occurence_map[bp] = []
                if byte_seq not in byte_pair_occurence_map[bp]:
                    byte_pair_occurence_map[bp].append(byte_seq)
        # 3.2 find the max count and max byte-pair
        max_count = max(list(byte_pair_map.values()))
        max_bp = None
        for bp, cnt in byte_pair_map.items():
            if cnt == max_count:
                max_bp = bp if max_bp == None or bp > max_bp else max_bp
        merges.append(max_bp)
        merged_bytes = max_bp[0] + max_bp[1]
        vocab[vocab_size] = merged_bytes
        vocab_size += 1
        # 3.3 replace the max_bp with the new bytes
        old_per_token_map = per_token_map
        per_token_map = {}
        for byte_seq, cnt in old_per_token_map.items():
            if byte_seq in byte_pair_occurence_map[max_bp]:
                i = 0
                new_byte_seq = ()
                while i < len(byte_seq):
                    if byte_seq[i] == max_bp[0] and (i+1) < len(byte_seq) and byte_seq[i+1] == max_bp[1]:
                        new_byte_seq += (merged_bytes,)
                        i += 2
                    else:
                        new_byte_seq += (byte_seq[i],)
                        i += 1
                per_token_map[new_byte_seq] = cnt
            else:
                per_token_map[byte_seq] = cnt
        # a b c d => (a,b) (b,c) (c,d) => max_bp = (b,c) => (a,bc) (bc,d)
        byte_pair_map = {}
        byte_pair_occurence_map = {}
        iteration += 1
    return vocab, merges


def run_fast(input_path: str,
        max_vocab_size: int,
        special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    trainer = BPETrainer()
    return trainer.train(input_path=input_path, max_vocab_size=max_vocab_size, special_tokens=special_tokens)


def save_results(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], corpus_name: str):
    from tests.common import gpt2_bytes_to_unicode
    decoder = gpt2_bytes_to_unicode()
    vocab0: dict[str, int] = {}
    for id, token in vocab.items():
        s = "".join([decoder[t] for t in token])
        vocab0[s] = id
    try:
        with open(f"{corpus_name}_vocab.json", 'w') as f:
            json.dump(vocab0, f, indent=4, ensure_ascii=False)
        with open(f"{corpus_name}_merges.txt", "w") as f:
            for merge in merges:
                f.write(f'{merge[0]} {merge[1]}\n')
    except Exception as e:
        print(f'Error when saving results: {e}')


if __name__ == "__main__":
    corpus = "owt_train"
    import time
    begin = time.time()
    # vocab1, merges1 = run(input_path=corpus, max_vocab_size=1000, special_tokens=["<|endoftext|>"])
    mid = time.time()
    vocab2, merges2 = run_fast(input_path=f"data/{corpus}.txt", max_vocab_size=32000, special_tokens=["<|endoftext|>"])
    end = time.time()
    print(f'Naive Run cost {(mid - begin):.3f}')
    print(f'Optimized Run cost {(end - mid):.3f}')
    save_results(vocab2, merges2, corpus_name=corpus)
