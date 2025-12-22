# Handout

## BPE Tokenization

- Train TinyStoriesV2-GPT4-train.txt dataset: cost 77.364 seconds
- Train owt_train.txt dataset: 13348.869 seconds (3.7 hours)

## Transformer

### Architecture

- Input: (batch_size, seq_len), and each token is a token_id
- Embedding Layer
    - A parameter matrix: shape=(vocab_size, d_model), and mapping the token_id to embedding
    - return (batch_size, seq_len, d_model)
    - computation: batch_size * seq_len * d_model
- Multi-Layer Transformer Block
    - Pre-Norm Causal-Masking Multi-head Self-Attention
        - RMSNorm
            - weight: (d_model,)
            - element-wise multiple
            - computation: 7 * batch_size * seq_len * d_model
        - Causal-Masking Multi-head Self-Attention
            - WQ/WK/WV: (num_heads\*d_k, d_model), WO: (d_model, num_heads\*d_v)
            - get Q/K/V: (batch_size, seq_len, d_model) => (batch_size, num_heads, seq_len, d_k/d_v)
                - computation: 3 * (2 * d_model * batch_size * seq_len * d_model)
            - ROPE(Q/K): (batch_size, num_heads, seq_len, d_k)
                - indexing + MM
                - computation: 2 * (batch_size * num_heads * seq_len * d_k * d_k + 2 * d_k * batch_size * num_heads * seq_len * d_k)
            - QK: (batch_size, num_heads, seq_len, d_k) * (batch_size, num_heads, seq_len, d_k) => (batch_size, num_heads, seq_len, seq_len)
                - computation: 2 * d_k * (batch_size * num_heads * seq_len * seq_len)
            - scaled & mask & softmax:
                - computation: 7 * (batch_size * num_heads * seq_len * seq_len)
            - (QK)*V: (batch_size, num_heads, seq_len, seq_len) => (batch_size, num_heads, seq_len, d_v)
                - computation: 2 * seq_len * batch_size * num_heads * seq_len * d_v
            - WO*(): (batch_size, seq_len, d_model)
                - computation: 2 * d_model * batch_size * seq_len * d_model
            - return (batch_size, seq_len, d_model)
        - Residual-Add
            - computation: batch_size * seq_len * d_model
    - Pre-Norm Position-wise Feed-Forward Network
        - RMSNorm: same
        - SwiGLU: SwiGLU(x, W1, W2, W3) = W2(SiLU(W1x) · (W3x)) = W2 *(((W1x)(sigmoid(W1x))) · (W3x))
            - W1/W3: (d_ff, d_model) W2: (d_model, d_ff)
            - W1x, W3x: (batch_size, seq_len, d_ff)
                - computation: 2 * (2 * d_model * batch_size * seq_len * d_ff)
            - SiLU(W1x)·(W3x): (batch_size, seq_len, d_ff)
                - computation: 6 * batch_size * seq_len * d_ff
            - W2*(): (batch_size, seq_len, d_model)
                - computation: 2 * d_ff * batch_size * seq_len * d_model
            - return (batch_size, seq_len, d_model)
        - Residual-Add
- Norm
    - (d_model,)
- Linear (d_model => vocab_size)
    - (vocab_size, d_model)
    - computation: 2 * d_model * batch_size * seq_len * vocab_size
- softmax (convert to a distribution)
    - computation: 5 * batch_size * seq_len * vocab_size


### Computation and Memory Analysis

- GPT-2 XL
    - 2,127,057,600 parameters and about 8 GB to load the model.
    - about 25314623488000 FLOPs.
        - embedding: 0.65%, attention(single layer): 1.81%, ffn(single layer): 0.24%, final linear: 0.65%
    - when context_length increase from 1024 to 16384, the ratio of attention computation(single layer) increate to **2.06%**, and almost all computation is used on attention.
- GPT-2 small(12-layers, 768-d_model, 12-heads)
    - embedding: 2.56%, attention(single layer): 6.93%, ffn(single layer): 0.98%, final linear: 2.56%
- GPT-2 medium(24-layers, 1024-d_model, 16-heads)
    - embedding: 1.30%, attention(single layer): 3.56%, ffn(single layer): 0.50%, final linear: 1.30%
- GPT-2 large(36-layers, 1280-d_model, 20-heads)
    - embedding: 0.87%, attention(single layer): 2.40%, ffn(single layer): 0.33%, final linear: 0.87%


## Training

### Learning rate

run step for 100 iterations
- lr=1: loss decrease from 26.27 to 12.51
- lr=1e1: loss decrease from 26.27 to 0.01
- lr=1e2: loss decrease from 26.27 to 0(at iter#67)
    - gradient vanishing
- lr=1e3: loss decrease from 26.27 to inf(keep increasing, reach inf at iter#25)
    - gradient explosion

## Training Experiment

0. about function code
- many functions have inefficient code, and lead to slow training
    - such as, indexing, it can be done by some simple torch api, but i use the naive solution.

1. encode corpus
- single process is slow, so we need to use multi-process to deal it
    - but I find that my host(20-cores, 5mins) is faster than the gpu server(128-cores, 16mins)
2. training speed
- the doc said that it takes about 1h and 22m to train 5000 steps in the cpu, but in my host, I need about 16h and 40m, and in the 4090 gpu server, it is also slow(maybe it's my code fault)
    - I need to find why?
3. about gradient vanish
- the reason is a bug in cross-entropy computation, use the logit_norm[target], not logit[target].
- Why?

4. extend exploration
- 