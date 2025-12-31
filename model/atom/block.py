import torch
import torch.nn as nn

from config.base import TransformerConfig
from model.atom.basic import MultiHeadAttention

class EncoderBlock(nn.Module):
    """
    Transformer encoder block with multi-head self-attention and feed-forward network.

    Applies multi-head self-attention followed by a position-wise feed-forward network, each wrapped with residual connections and layer normalization.

    Args:
        config (GrootTinyConfig): Model configuration containing input, attention, and transformer hyperparameters.

    Returns:
        torch.FloatTensor: Output tensor of shape (batch_size, seq_len, embedding_dim).
    """
    def __init__(self, config: TransformerConfig):
        super(EncoderBlock, self).__init__()

        self.config = config

        self.embedding_dim  = self.config.input.embedding_dim
        self.fnn_factor     = self.config.transformer.fnn_factor
        self.num_heads      = self.config.attention.num_heads
        self.device         = self.config.device
        self.dropout_prob   = self.config.transformer.dropout_prob
        self.init           = self.config.init

        self.multi_head_attention = MultiHeadAttention(config=self.config, mask=False)
        self.normalization_attn = nn.LayerNorm(self.embedding_dim)
        self.normalization_fnn = nn.LayerNorm(self.embedding_dim)
        self.fnn = nn.Sequential(
            nn.Linear(in_features=self.embedding_dim, out_features=self.embedding_dim * self.fnn_factor),
            nn.GELU(), # TODO: Try other activations, GeLU, SiLU, Swish
            nn.Linear(in_features=self.embedding_dim * self.fnn_factor, out_features=self.embedding_dim),
            nn.Dropout(self.dropout_prob)
        )
        
        self.fnn.apply(self._init_weights)
    
    def _init_weights(self, module):
        if type(module) == nn.Linear:
            torch.nn.init.normal_(module.weight, mean=self.init.linear_mean, std=self.init.linear_std)

    def forward(self, x):
        # TODO: Current: MHSA -> Layer Norm -> FNN -> Layer Norm, Have to try: Layer Norm -> MHSA -> Layer Norm -> FNN 
        mhsa_pre_norm = self.normalization_attn(x)
        mhsa = self.multi_head_attention(mhsa_pre_norm, mhsa_pre_norm, mhsa_pre_norm)
        mhsa_output = (mhsa + x)

        fnn_pre_norm = self.normalization_fnn(mhsa_output)
        fnn = self.fnn(fnn_pre_norm)
        fnn_output = (fnn + mhsa_output)

        return fnn_output
    
class DecoderBlock(nn.Module):
    """
    Transformer decoder block with multi-head masked self-attention and cross-attention.

    Applies causal (masked) self-attention, encoder–decoder cross-attention, and a position-wise feed-forward network, each with residual connections and layer normalization.

    Args:
        config (GrootTinyConfig): Model configuration containing input, attention, and transformer hyperparameters.

    Returns:
        torch.FloatTensor: Output tensor of shape (batch_size, seq_len, embedding_dim).
    """
    def __init__(self, config: TransformerConfig):
        super(DecoderBlock, self).__init__()

        self.config = config

        self.embedding_dim  = self.config.input.embedding_dim
        self.fnn_factor     = self.config.transformer.fnn_factor
        self.num_heads      = self.config.attention.num_heads
        self.device         = self.config.device
        self.dropout_prob   = self.config.transformer.dropout_prob
        self.init           = self.config.init

        self.masked_multi_head_attention = MultiHeadAttention(config=self.config, mask=True)
        self.normalization_mhsa = nn.LayerNorm(self.embedding_dim)

        # self.cross_attention = MultiHeadAttention(config=self.config, mask=False).to(self.device)
        # self.normalization_cross_mhsa = nn.LayerNorm(self.embedding_dim).to(self.device)

        self.fnn = nn.Sequential(
            nn.Linear(in_features=self.embedding_dim, out_features=self.embedding_dim * self.fnn_factor),
            nn.GELU(),
            nn.Linear(in_features=self.embedding_dim * self.fnn_factor, out_features=self.embedding_dim),
            nn.Dropout(self.dropout_prob)
        )
        self.normalization_fnn = nn.LayerNorm(self.embedding_dim)

        self.fnn.apply(self.__init_weights)
    
    def __init_weights(self, module):
        if type(module) == nn.Linear:
            torch.nn.init.normal_(module.weight, mean=self.init.linear_mean, std=self.init.linear_std)
    
    def forward(self, x, v_cross, k_cross):
        masked_mhsa_pre_norm = self.normalization_mhsa(x)
        masked_mhsa = self.masked_multi_head_attention(masked_mhsa_pre_norm, masked_mhsa_pre_norm, masked_mhsa_pre_norm)
        masked_mhsa_output = masked_mhsa + x 

        # cross_mhsa_pre_norm = self.normalization_cross_mhsa(masked_mhsa_output)
        # cross_mhsa = self.cross_attention(cross_mhsa_pre_norm, k_cross, v_cross)
        # cross_mhsa_output = cross_mhsa + masked_mhsa_output

        fnn_pre_norm = self.normalization_fnn(masked_mhsa_output)
        fnn = self.fnn(fnn_pre_norm)
        fnn_output = masked_mhsa_output + fnn

        return fnn_output

class TransformerEncoder(nn.Module):
    def __init__(self, config: TransformerConfig):
        super(TransformerEncoder, self).__init__()

        self.config = config
        self.num_encoders   = self.config.transformer.num_encoders

        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(config=self.config) for _ in range(self.num_encoders)
        ])
    
    def forward(self, x):
        for block in self.encoder_blocks:
            x = block(x)
        return x
    
class TransformerDecoder(nn.Module):
    def __init__(self, config: TransformerConfig):
        super(TransformerDecoder, self).__init__()

        self.config = config
        self.num_decoders   = self.config.transformer.num_decoders

        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(config=self.config) for _ in range(self.num_decoders)
        ])
    
    def forward(self, x):
        for block in self.decoder_blocks:
            x = block(x, x, x)
        return x