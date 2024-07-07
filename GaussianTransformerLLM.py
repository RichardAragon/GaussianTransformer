import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussianTransformerLLM(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        super(GaussianTransformerLLM, self).__init__()
        self.embedding = ProbabilisticEmbedding(vocab_size, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True),
            num_layers
        )
        self.gaussian_layer = GaussianLayer(d_model)
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, src, src_mask=None):
        embedded = self.embedding(src)
        transformer_output = self.transformer(embedded, src_mask)
        gaussian_output, uncertainty = self.gaussian_layer(transformer_output)
        combined_output = self.combine_outputs(transformer_output, gaussian_output, uncertainty)
        return self.output_layer(combined_output), uncertainty

    def combine_outputs(self, transformer_output, gaussian_output, uncertainty):
        confidence = 1 - uncertainty
        return confidence * gaussian_output + (1 - confidence) * transformer_output

class ProbabilisticEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, noise_std=0.1):
        super(ProbabilisticEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.noise_std = noise_std

    def forward(self, x):
        embedded = self.embedding(x)
        if self.training:
            noise = torch.randn_like(embedded) * self.noise_std
            return embedded + noise
        return embedded

class GaussianLayer(nn.Module):
    def __init__(self, d_model):
        super(GaussianLayer, self).__init__()
        self.mean_proj = nn.Linear(d_model, d_model)
        self.var_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        mean = self.mean_proj(x)
        log_var = self.var_proj(x)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        sample = mean + eps * std
        uncertainty = torch.mean(log_var, dim=-1)
        return sample, uncertainty

# Gaussian attention mechanism
def gaussian_attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    gaussian_weights = torch.exp(-scores ** 2 / 2)
    p_attn = F.softmax(gaussian_weights, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

# Usage example
vocab_size = 10000
d_model = 512
nhead = 8
num_layers = 6
dim_feedforward = 2048

model = GaussianTransformerLLM(vocab_size, d_model, nhead, num_layers, dim_feedforward)

# Example input (batch_size=32, sequence_length=50)
src = torch.randint(0, vocab_size, (32, 50))
output, uncertainty = model(src)

print(f"Output shape: {output.shape}")
print(f"Uncertainty shape: {uncertainty.shape}")
