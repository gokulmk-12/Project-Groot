import torch
import torch.nn as nn
import torch.nn.functional as F

from model.atom.basic import InputEmbedding
from config.base import TransformerConfig
from model.residual.mhc import mHCResidual
from model.atom.block import TransformerEncoder, TransformerDecoder

class Transformer(nn.Module):
    """
    Encoder–decoder Transformer model for sequence modeling and generation.

    Embeds input tokens, processes them through stacked encoder and decoder blocks, and projects outputs to vocabulary logits.

    Args:
        config (GrootTinyConfig): Model configuration defining input, attention, and transformer hyperparameters.

    Returns:
        Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]: Vocabulary logits and (optionally) cross-entropy loss.
    """
    def __init__(self, config: TransformerConfig):
        super(Transformer, self).__init__()

        self.config = config

        self.context_length = self.config.input.context_length
        self.vocab_size     = self.config.input.vocab_size
        self.embedding_dim  = self.config.input.embedding_dim
        self.num_encoders   = self.config.transformer.num_encoders
        self.num_decoders   = self.config.transformer.num_decoders
        self.device         = self.config.device
        self.init           = self.config.init

        self.input_embedding = InputEmbedding(config=config)
        # self.encoder = TransformerEncoder(config=config)
        self.decoder = TransformerDecoder(config=config)

        self.classification_layer = nn.Linear(in_features=self.embedding_dim, out_features=self.vocab_size)

        if self.config.transformer.allow_mhc:
            self.final_mhc = mHCResidual(dim=self.embedding_dim, n_streams=self.config.transformer.n_streams)
            self.final_ln = nn.LayerNorm(self.embedding_dim)
        
    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params 
    
    def forward(self, x: torch.LongTensor, targets: torch.FloatTensor = None):
        x_embed = self.input_embedding(x)

        if self.config.transformer.allow_mhc:
            x_embed = x_embed.unsqueeze(2).expand(-1, -1, self.config.transformer.n_streams, -1)
        
        # encoder_out = self.encoder.forward(x_embed)
        decoder_out = self.decoder.forward(x_embed)

        if self.config.transformer.allow_mhc:
            decoder_out = self.final_mhc.get_aggregated_input(decoder_out)
            decoder_out = self.final_ln(decoder_out)

        logits = self.classification_layer(decoder_out)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1)
            )
        else:
            loss = None
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, x: torch.LongTensor, max_new_tokens: int, temperature: int = 1.0, top_k: int = None, top_p: float = None):
        eos_token_id = 50256

        for _ in range(max_new_tokens):
            x_new = x[:, -self.context_length:]
            logits, _ = self(x_new)

            # Temperature Scaling
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)

            # Top-P Sampling
            if top_p is not None:
                sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
                cum_probs = torch.cumsum(sorted_probs, dim=-1)
                top_p_mask = cum_probs <= top_p
                top_p_mask[..., 0] = True
                top_p_probs = sorted_probs * top_p_mask
                top_p_probs = top_p_probs / top_p_probs.sum(dim=-1, keepdim=True)

                sample_idx = torch.multinomial(top_p_probs, num_samples=1)
                x_next = torch.gather(sorted_indices, -1, sample_idx)

            # Top-K Sampling
            elif top_k is not None:
                top_k = min(top_k, probs.size(-1))
                top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
                
                top_k_probs = top_k_probs / top_k_probs.sum(keepdim=True, dim=-1)
                sample_idx = torch.multinomial(top_k_probs, num_samples=1)
                x_next = torch.gather(top_k_indices, -1, sample_idx)
            
            # Greedy Decoding
            else:
                x_next = torch.argmax(probs, dim=-1, keepdim=True)

            if x_next.squeeze(-1) == eos_token_id:
                break  
            x = torch.cat((x, x_next), dim=1)    
        return x