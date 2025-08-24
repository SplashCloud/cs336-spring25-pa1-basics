import regex as re


'''
A example to show how BPE works
'''

# PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

corpus = '''low low low low low
lower lower widest widest widest
newest newest newest newest newest newest
<|endoftext|>
'''

special_tokens = [b'<|endoftext|>']

# 1. Vocabulary Initialization
voc_size = 256
vocabulary: dict[bytes, int] = {}
for i in range(voc_size):
    vocabulary[bytes([i])] = i
for st in special_tokens:
    vocabulary[st] = voc_size
    voc_size += 1
print(f'Vocabulary Size = {voc_size} after initialization')

# 2. Per-Tokenization
content = corpus.split() # simple way to split the corpus with '\t', '\n', ' '
per_token_map: dict[tuple[bytes], int] = {}
for token in content:
    encoded_token = token.encode()
    if encoded_token in special_tokens:
        continue
    byte_seq = tuple(bytes([b]) for b in list(encoded_token))
    if byte_seq not in per_token_map.keys():
        per_token_map[byte_seq] = 0
    per_token_map[byte_seq] += 1

print(per_token_map)

# 3. Merge Byte Pair (Training)
# Think how to write the logic
byte_pair_map: dict[tuple[bytes, bytes], int] = {}
byte_pair_occurence_map: dict[tuple[bytes, bytes], list[tuple[bytes]]] = {} # record the mapping from byte-pair to origin byte sequence
iteration = 0
max_iteration = 6
while iteration < max_iteration: # there may be a voc_size threshold
    print(f'BPE Training Itration#{iteration} Begins...')
    # 3.1 count all the byte-pair within all tokens
    for byte_seq, cnt in per_token_map.items():
        for i in range(len(byte_seq)-1):
            bp = tuple((byte_seq[i], byte_seq[i+1]))
            if bp not in byte_pair_map.keys():
                byte_pair_map[bp] = 0
            byte_pair_map[bp] += cnt
            if bp not in byte_pair_occurence_map.keys():
                byte_pair_occurence_map[bp] = []
            if byte_seq not in byte_pair_occurence_map[bp]:
                byte_pair_occurence_map[bp].append(byte_seq)
    print(byte_pair_map)
    # 3.2 find the max_bp with the max_cnt
    max_count = max(list(byte_pair_map.values()))
    max_bp = None
    for bp, cnt in byte_pair_map.items():
        if cnt == max_count:
            max_bp = bp if max_bp is None or bp > max_bp else max_bp
    print(f'Select to merge {max_bp} byte-pair with {max_count} occurences')
    # 3.3 replace the selected bp with merged one
    old_per_token_map = per_token_map
    per_token_map = {}
    for byte_seq, cnt in old_per_token_map.items():
        if byte_seq in byte_pair_occurence_map[max_bp]:
            new_byte_seq = ()
            i = 0
            while i < len(byte_seq):
                if max_bp[0] == byte_seq[i] and (i+1) < len(byte_seq) and max_bp[1] == byte_seq[i+1]:
                    new_byte_seq += (max_bp[0] + max_bp[1],)
                    i += 2
                else:
                    new_byte_seq += (byte_seq[i],)
                    i += 1
            per_token_map[new_byte_seq] = cnt
        else:
            per_token_map[byte_seq] = cnt
    vocabulary[max_bp[0] + max_bp[1]] = voc_size
    print(f'Assign the id#{voc_size} to the merged bp {max_bp[0] + max_bp[1]}')
    voc_size += 1
    # clear dict
    byte_pair_map = {}
    byte_pair_occurence_map = {}
    print(per_token_map)
    print(f'BPE Training Itration#{iteration} Ends...')
    iteration += 1

# check final vocabulary
for byte, id in vocabulary.items():
    print(byte, end=',')
print()