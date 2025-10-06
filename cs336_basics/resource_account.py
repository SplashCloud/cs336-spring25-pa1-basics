class MemoryAccounting:

    def __init__(self, vocab_size: int, d_model: int, d_ff: int, num_heads: int, num_layers: int, batch_size: int, context_length: int):
        # model
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.num_layers = num_layers
        # input
        self.batch_size = batch_size
        self.context_length = context_length

    def rmsnorm(self):
        return self.d_model

    def embedding(self):
        return self.vocab_size * self.d_model

    def attention(self):
        return 4 * self.d_model * self.d_model

    def ffn(self):
        return 3 * self.d_ff * self.d_model

    def final_linear(self):
        return self.vocab_size * self.d_model

    def model(self):
        total = self.embedding() + self.num_layers * (self.rmsnorm() + self.attention() + self.rmsnorm() + self.ffn()) + self.rmsnorm() + self.final_linear()
        return total

    def activation(self):
        ''' need to store the activation to calculate grad(backpropagation) '''
        hidden1 = self.batch_size * self.context_length * self.d_model
        hidden2 = self.batch_size * self.context_length * self.d_ff
        output = self.batch_size * self.context_length * self.vocab_size
        attn_activation = 7 * hidden1
        ffn_activation = 4 * hidden1 + hidden2
        return self.num_layers * (hidden1 + attn_activation + hidden1 + ffn_activation) + hidden1 + output + 10 * output

    def adamw(self):
        grad = self.model() - self.embedding() # embedding parameters are not learnable
        w = v = grad
        return w + v


class ComputationAccounting:

    def __init__(self, vocab_size: int, d_model: int, d_ff: int, num_heads: int, num_layers: int, batch_size: int, context_length: int):
        # model
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.num_layers = num_layers
        # input
        self.batch_size = batch_size
        self.context_length = context_length

    def embedding(self):
        return self.batch_size * self.context_length * self.d_model

    def rmsnorm(self):
        return 7 * self.batch_size * self.context_length * self.d_model

    def attention(self):

        d_k = d_v = self.d_model // self.num_heads

        def Q_K_V():
            ''' (bs, seq_len, d_model) * (d_model, d_model) = (bs, seq_len, d_model) for Q,K,V '''
            return 3 * (2 * self.d_model * (self.batch_size * self.vocab_size * self.d_model))

        def RoPE():
            indexing = self.batch_size * self.num_heads * self.context_length * d_k * d_k
            matrix_multiply = 2 * d_k * self.batch_size * self.num_heads * self.context_length * d_k
            return indexing + matrix_multiply
        
        def QK():
            ''' (bs, heads, seq_len, d_k) * (bs, heads, seq_len, d_k) = (bs, heads, seq_len, seq_len) '''
            return 2 * d_k * (self.batch_size * self.num_heads * self.context_length * self.context_length)

        def scale_mask_softmax():
            return 7 * (self.batch_size * self.num_heads * self.context_length * self.context_length)
        
        def QKV():
            ''' (bs, heads, seq_len, seq_len) * (bs, heads, seq_len, d_v) = (bs, heads, seq_len, d_v) '''
            return 2 * self.context_length * (self.batch_size * self.num_heads * self.context_length * d_v)

        def Output():
            ''' (bs, seq_len, d_model) * (d_model, d_model) = (bs, seq_len, d_model) '''
            return 2 * self.d_model * (self.batch_size * self.vocab_size * self.d_model)

        return Q_K_V() + 2 * RoPE() + QK() + scale_mask_softmax() + QKV() + Output()

    def ffn(self):
        def Wx():
            ''' (bs, seq_len, d_model) * (d_ff, d_model) = (bs, seq_len, d_ff) '''
            return 2 * self.d_model * (self.batch_size * self.context_length * self.d_ff)
        
        def element_wise_compt():
            return  6 * self.batch_size * self.context_length * self.d_ff
        
        return 3 * Wx() + element_wise_compt()

    def final_linear(self):
        ''' (bs, seq_len, d_model) * (vocab_size, d_model) = (bs, seq_len, vocab_size) '''
        return 2 * self.d_model * (self.batch_size * self.context_length * self.vocab_size)

    def softmax(self):
        return 5 * self.batch_size * self.context_length * self.vocab_size

    def model(self):
        total = self.embedding() + self.num_layers * (self.rmsnorm() + self.attention() + self.rmsnorm() + self.ffn()) + self.rmsnorm() + self.final_linear() + self.softmax()
        return total

    def adamw(self, model_size):
        N = 13 # for every parameter, need 13 FLOPs
        return N * model_size


class ResourceAccounting:

    def __init__(self, vocab_size: int, d_model: int, d_ff: int, num_heads: int, num_layers: int, batch_size: int, context_length: int):
        self.memory_account = MemoryAccounting(vocab_size=vocab_size, d_model=d_model, d_ff=d_ff, num_heads=num_heads, num_layers=num_layers, batch_size=batch_size, context_length=context_length)
        self.computation_account = ComputationAccounting(vocab_size=vocab_size, d_model=d_model, d_ff=d_ff, num_heads=num_heads, num_layers=num_layers, batch_size=batch_size, context_length=context_length)

    
    def calculate_memory(self, model_params: int, dtype: str = "float32"):
        datasize = 4 if dtype == "float32" else 1
        return model_params * datasize / 1024 / 1024 / 1024

    
    def report(self):
        model_params = self.memory_account.model()
        memory_cost = self.calculate_memory(model_params)
        print("===== Memory Report =====")
        print(f"Model parameters: {model_params}")
        print(f"Model parameters cost {memory_cost} GB")
        print(f"Model gradients cost {memory_cost} GB")
        print(f"Model activation cost {self.calculate_memory(self.memory_account.activation())} GB")
        print(f"AdamW Optimizer need memory: {self.calculate_memory(self.memory_account.adamw())} GB")
        print("===== Computation Report =====")
        print(f"Model run a forward need {self.computation_account.model()} FLOPs")
        print(f"AdamW run a step need {self.computation_account.adamw(model_params)} FLOPs")



if __name__ == "__main__":
    vocab_size = 50257
    batch_size = 2
    context_length = 1024
    d_ff = 6400
    data_size = 4

    # GPT-2 XL
    num_layers = 48
    d_model = 1600
    num_heads = 25
    RA = ResourceAccounting(vocab_size=vocab_size, d_model=d_model, d_ff=d_ff, num_heads=num_heads, num_layers=num_layers, batch_size=batch_size, context_length=context_length)
    RA.report()