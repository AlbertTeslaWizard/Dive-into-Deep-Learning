import math
import torch
from torch import nn
import matplotlib.pyplot as plt

def sequence_mask(X, valid_lens, value=0):
    """Masks entries in a sequence with a given value.

    Args:
        X (torch.Tensor): The input tensor.
        valid_lens (torch.Tensor): A tensor containing the valid lengths of sequences.
        value (float, optional): The value to use for masking. Defaults to 0.

    Returns:
        torch.Tensor: The masked tensor.
    """
    maxlen = X.size(1)
    mask = torch.arange(maxlen, dtype=torch.float32, device=X.device)[None, :] < valid_lens[:, None]
    X[~mask] = value
    return X

def masked_softmax(X, valid_lens):
    """
    Performs softmax operation by masking elements on the last axis.

    Args:
        X (torch.Tensor): A 3D tensor.
        valid_lens (torch.Tensor): A 1D or 2D tensor containing valid lengths.

    Returns:
        torch.Tensor: The result of the softmax operation.
    """
    # If no valid_lens are provided, perform a regular softmax
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        # If valid_lens is a 1D tensor, repeat it to match the batch size
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        
        # Mask the elements that are beyond the valid lengths
        # by replacing them with a very large negative value.
        # This ensures their softmax output is close to 0.
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)

class AdditiveAttention(nn.Module):
    """
    Additive Attention module.
    """
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.attention_weights = None

    def forward(self, queries, keys, values, valid_lens):
        # Apply linear transformations to queries and keys
        queries, keys = self.W_q(queries), self.W_k(keys)
        
        # Expand dimensions for broadcasting
        # queries shape: (batch_size, no. of queries, 1, num_hiddens)
        # keys shape: (batch_size, 1, no. of key-value pairs, num_hiddens)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        
        # Calculate attention scores
        # scores shape: (batch_size, no. of queries, no. of key-value pairs)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        
        # Apply dropout to attention weights and compute the context vector
        # values shape: (batch_size, no. of key-value pairs, value dimension)
        return torch.bmm(self.dropout(self.attention_weights), values)

def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5), cmap='Reds'):
    """
    Displays heatmaps of matrices.
    """
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize, sharex=True, sharey=True, squeeze=False)

    for i, row_axes in enumerate(axes):
        for j, ax in enumerate(row_axes):
            pcm = ax.imshow(matrices[i, j].detach().numpy(), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6)


# --- Additive Attention Example ---
queries, keys = torch.normal(0, 1, (2, 1, 20)), torch.ones((2, 10, 2))
# The values tensor is created and repeated for the batch
values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(2, 1, 1)
valid_lens = torch.tensor([2, 6])

# Instantiate and run the AdditiveAttention model
attention = AdditiveAttention(key_size=2, query_size=20, num_hiddens=8, dropout=0.1)
attention.eval()
attention(queries, keys, values, valid_lens)

# Visualize the attention weights
show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)),
                  xlabel='Keys', ylabel='Queries')
plt.show()


class DotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention module.
    """
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.attention_weights = None

    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        
        # Calculate scores using batch matrix multiplication and scale by sqrt(d)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        
        # Apply dropout to attention weights and compute the context vector
        return torch.bmm(self.dropout(self.attention_weights), values)


# --- Dot-Product Attention Example ---
queries = torch.normal(0, 1, (2, 1, 2))

# Instantiate and run the DotProductAttention model
attention = DotProductAttention(dropout=0.5)
attention.eval()
print(attention(queries, keys, values, valid_lens))

# Visualize the attention weights
show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)),
                  xlabel='Keys', ylabel='Queries')
plt.show()