import torch
import numpy as np
import torch.nn as nn

from config.base import TransformerConfig

class InputEmbedding(nn.Module):
    """
    Adds token and positional embeddings for an input seqeunce

    Args:
        config (GrootTinyConfig): Model configuration containing input, attention, and transformer hyperparameters.
    
    Forward Args:
        token_ids (torch.LongTensor): Tensor of shape (batch_size, seq_len) containing token indices.
    
    Returns:
        torch.FloatTensor: Input Embeddings of shape (batch_size, seq_len, embedding_dim)
    """
    def __init__(self, config: TransformerConfig):
        super(InputEmbedding, self).__init__()

        self.config = config
        input_config = self.config.input
        self.embedding_layer = nn.Embedding(num_embeddings=input_config.vocab_size, embedding_dim=input_config.embedding_dim)
        self.pos_embedding_layer = nn.Embedding(num_embeddings=input_config.context_length, embedding_dim=input_config.embedding_dim)
        self.context_length = input_config.context_length

        torch.nn.init.normal_(self.embedding_layer.weight, mean=0, std=0.02)
        torch.nn.init.normal_(self.pos_embedding_layer.weight, mean=0, std=0.02)
    
    def forward(self, token_ids: torch.LongTensor):
        _, seq_len = token_ids.shape
        token_embedding = self.embedding_layer(token_ids)
        pos_indices = torch.arange(seq_len).to(self.config.device)
        pos_embedding = self.pos_embedding_layer(pos_indices).unsqueeze(0)
        total_embedding = pos_embedding + token_embedding
        return total_embedding
    
class MultiHeadAttention(nn.Module):
    """
    Multi-head scaled dot-product attention module

    Args:
        config (GrootTinyConfig): Model configuration containing input, attention, and transformer hyperparameters.
    
    Forward Args:
        q, k, v (torch.FloatTensor): Tensors of shape (batch_size, seq_len, embedding_dim)
    
    Returns:
        torch.FloatTensor: Output tensor of shape (batch_size, seq_len, embedding_dim)
    """
    def __init__(self, config: TransformerConfig, mask: bool = False):
        super(MultiHeadAttention, self).__init__()
        self.config = config

        self.embedding_dim  = self.config.input.embedding_dim
        self.num_heads      = self.config.attention.num_heads
        self.head_dim       = self.embedding_dim // self.num_heads
        self.context_length = self.config.input.context_length
        self.mask           = mask
        self.device         = self.config.device
        self.init           = self.config.init
        self.dropout_prob   = self.config.transformer.dropout_prob

        self.W_q = nn.Linear(in_features=self.embedding_dim, out_features=self.embedding_dim, bias=False) # Query
        self.W_k = nn.Linear(in_features=self.embedding_dim, out_features=self.embedding_dim, bias=False) # Key
        self.W_v = nn.Linear(in_features=self.embedding_dim, out_features=self.embedding_dim, bias=False) # Value

        self.output = nn.Linear(in_features=self.embedding_dim, out_features=self.embedding_dim)
        num_layers = self.config.transformer.num_encoders + self.config.transformer.num_decoders
        scaled_std = self.init.linear_std / np.sqrt(2 * num_layers)

        torch.nn.init.normal_(self.W_q.weight, mean=self.init.linear_mean, std=self.init.linear_std)
        torch.nn.init.normal_(self.W_k.weight, mean=self.init.linear_mean, std=self.init.linear_std)
        torch.nn.init.normal_(self.W_v.weight, mean=self.init.linear_mean, std=self.init.linear_std)
        
        torch.nn.init.normal_(self.output.weight, mean=self.init.linear_mean, std=scaled_std)

        self.attn_dropout   = nn.Dropout(p=self.dropout_prob)
        self.resid_dropout  = nn.Dropout(p=self.dropout_prob)
    
    def forward(self, q: torch.FloatTensor, k: torch.FloatTensor, v: torch.FloatTensor):
        batch, seq_len, _ = q.shape
        Q = self.W_q(q)
        K = self.W_k(k)
        V = self.W_v(v)
        Q_split, K_split, V_split = self.split(Q, K, V)

        attention, _ = self.scaled_dot_product_attention(Q_split, K_split, V_split)
        attention_swap = attention.transpose(1, 2)
        full_attention = attention_swap.reshape(batch, seq_len, -1)

        mha_output = self.resid_dropout(self.output(full_attention))
        return mha_output
    
    def split(self, Q, K, V):
        Q_split = torch.stack(torch.split(Q, self.head_dim, dim=-1), dim=1)
        K_split = torch.stack(torch.split(K, self.head_dim, dim=-1), dim=1)
        V_split = torch.stack(torch.split(V, self.head_dim, dim=-1), dim=1)
        return Q_split, K_split, V_split
    
    def scaled_dot_product_attention(self, Q, K, V):
        batch, num_heads, seq_len, _ = Q.shape
        K_T = torch.transpose(K, -2, -1)
        QK_T = torch.einsum('abij, abjk -> abik', [Q, K_T])

        if self.mask:
            mask = torch.tril(torch.ones((seq_len, seq_len))).expand(batch, num_heads, seq_len, seq_len).to(self.device)
            QK_T = QK_T.masked_fill(mask==0, float('-inf'))
        
        d_k = K.shape[-1]
        scaled_product = QK_T / np.sqrt(d_k)
        attention_weights = torch.softmax(scaled_product, dim=-1)
        attention_weights = self.attn_dropout(attention_weights)
        attention = torch.einsum('abij, abjk -> abik', [attention_weights, V])

        return attention, attention_weights