import torch
import torch.nn as nn

from config.base import TransformerConfig
from model.atom.basic import MultiHeadAttention
from model.residual.mhc import mHCResidual

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
        """
        Currently does Pre-Norm, which is good for training stability and alleviates vanishing gradient, but leads to representation collapse, losing diversity and leading to identical learning. Post-Norm addresses representation collapse, but reintroduces vanishing gradients.
        """
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
        self.allow_mhc      = self.config.transformer.allow_mhc
        self.n_streams      = self.config.transformer.n_streams

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

        if self.allow_mhc:
            self.mhc_attn = mHCResidual(dim=self.embedding_dim, n_streams=self.n_streams)
            self.mhc_ffnn = mHCResidual(dim=self.embedding_dim, n_streams=self.n_streams)

        self.fnn.apply(self.__init_weights)
    
    def __init_weights(self, module):
        if type(module) == nn.Linear:
            torch.nn.init.normal_(module.weight, mean=self.init.linear_mean, std=self.init.linear_std)
    
    def forward(self, x, v_cross, k_cross):
        if self.allow_mhc:
            x_agg = self.mhc_attn.get_aggregated_input(x)
            masked_mhsa_pre_norm = self.normalization_mhsa(x_agg)
            masked_mhsa_output = self.masked_multi_head_attention(masked_mhsa_pre_norm, masked_mhsa_pre_norm, masked_mhsa_pre_norm)
            x = self.mhc_attn(x, masked_mhsa_output)

            x_agg = self.mhc_ffnn.get_aggregated_input(x)
            fnn_pre_norm = self.normalization_fnn(x_agg)
            fnn_output = self.fnn(fnn_pre_norm)
            x = self.mhc_ffnn(x, fnn_output)

            return x
        
        else:
            masked_mhsa_pre_norm = self.normalization_mhsa(x)
            masked_mhsa_output = self.masked_multi_head_attention(masked_mhsa_pre_norm, masked_mhsa_pre_norm, masked_mhsa_pre_norm)
            if not self.config.transformer.allow_mhc:
                masked_mhsa_output = masked_mhsa_output + x 

            # cross_mhsa_pre_norm = self.normalization_cross_mhsa(masked_mhsa_output)
            # cross_mhsa = self.cross_attention(cross_mhsa_pre_norm, k_cross, v_cross)
            # cross_mhsa_output = cross_mhsa + masked_mhsa_output

            fnn_pre_norm = self.normalization_fnn(masked_mhsa_output)
            fnn_output = self.fnn(fnn_pre_norm)
            if not self.config.transformer.allow_mhc:
                fnn_output = masked_mhsa_output + fnn_output

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