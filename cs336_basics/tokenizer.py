import os
import regex as re
import json
from typing import Iterable
from cs336_basics.logger import LoggerManager


logger = LoggerManager("TokenizerLogger")

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class BPETokenizer:

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None):
        self.decode_vocab = vocab
        self.encode_vocab: dict[bytes, int] = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.special_tokens = special_tokens
        if self.special_tokens:
            self.deal_with_special_tokens()


    def deal_with_special_tokens(self):
        assert self.special_tokens is not None
        # the special_token may include another one, sunch ["xx", "x"]
        # one solution is sort the special_tokens by length
        self.special_tokens = sorted(self.special_tokens, key=lambda x : len(x), reverse=True)
        delimiter = "|".join([re.escape(special_token) for special_token in self.special_tokens])
        self.delimiter = f'({delimiter})' # using capturing parentheses to keep special token in the result
        # add into vocab if exist new special tokens
        for special_token in self.special_tokens:
            encoded_st = special_token.encode(encoding="utf-8")
            if encoded_st not in self.encode_vocab.keys():
                self.encode_vocab[encoded_st] = len(self.encode_vocab)
                self.decode_vocab[len(self.decode_vocab)] = encoded_st


    @staticmethod
    def from_file(vocab_file: str, merges_file: str, special_tokens: list[str] | None):
        from tests.common import gpt2_bytes_to_unicode
        gpt2_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()} # visible str => int within [0, 255]
        with open(vocab_file, encoding="utf-8") as f:
            data = json.load(f)
            vocab = {
                vocab_idx: bytes([gpt2_decoder[token] for token in vocab_str])
                for vocab_str, vocab_idx in data.items()
            }
        with open(merges_file, encoding="utf-8") as f:
            merges_pair = [tuple(line.rstrip().split(" ")) for line in f.readlines()]
            merges = [
                (
                    bytes([gpt2_decoder[token] for token in merge_item1]),
                    bytes([gpt2_decoder[token] for token in merge_item2])
                ) 
                for merge_item1, merge_item2 in merges_pair
            ]
        return BPETokenizer(vocab, merges, special_tokens)


    def encode(self, text: str) -> list[int]:
        return self._encode(text)


    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for string in iterable:
            result = self._encode(string)
            for id in result:
                yield id


    def encode_file(self, filename: str) -> list[int]:
        chunk_size = 4096
        result = []
        
        with open(filename, "r") as f:
            f.seek(os.SEEK_END)
            file_size = f.tell()
            chunk_size = max(chunk_size, file_size//100)
            logger.info(f"Read {chunk_size} bytes per time from {filename}")
            offset = 0
            while True:
                f.seek(offset)
                content = f.read(chunk_size)
                offset += chunk_size
                if content == b'':
                    break
                if content[-1] != ' ': # cut the middle of the token
                    c = content[-1]
                    while c != ' ':
                        c = f.read(1)
                        content += c
                        offset += 1
                result.extend(self._encode(content))
        return result
                
    
    def _encode(self, content: str) -> list[int]:
        chunks = re.split(pattern=self.delimiter, string=content) if self.special_tokens else [content]
        result = []
        for chunk in chunks:
            if self.special_tokens and chunk in self.special_tokens:
                result.append(self.encode_vocab[chunk.encode(encoding="utf-8")])
                continue
            iter = re.finditer(pattern=PAT, string=chunk)
            for i in iter:
                # print(f'{i.group()} ', end='')
                token = i.group().encode(encoding="utf-8")
                if token in self.encode_vocab.keys():
                    # fast path for which is already in the vocab
                    result.append(self.encode_vocab[token])
                else:
                    merged_bytes_seq = self.merge(token)
                    encoded_seq = [self.encode_vocab[b] for b in merged_bytes_seq]
                    result.extend(encoded_seq)
        return result


    def merge(self, token: bytes) -> list[bytes]:
        '''
        apply the merges into per-tokens *in the same order of creation*
        '''
        bytes_sequence = [bytes([b]) for b in token]
        for merge in self.merges:
            i = 0
            merged_seq = []
            while i < len(bytes_sequence) - 1:
                item1, item2 = bytes_sequence[i], bytes_sequence[i+1]
                if item1 == merge[0] and item2 == merge[1]:
                    merged_seq.append(item1 + item2)
                    i += 2
                else:
                    merged_seq.append(item1)
                    i += 1
            while i < len(bytes_sequence):
                merged_seq.append(bytes_sequence[i])
                i += 1
            bytes_sequence = merged_seq
        return bytes_sequence


    def decode(self, encoded_seq: list[int]) -> str:
        result = b''
        for encoded_number in encoded_seq:
            result += self.decode_vocab[encoded_number]
        return result.decode(encoding="utf-8", errors="replace")
