import torch
from config.base import TransformerConfig

class GrootSmallConfig(TransformerConfig):
    name: str = "small"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    class input:
        vocab_size: int     = 50257
        context_length: int = 512
        batch_size: int     = 10
        embedding_dim: int  = 1024
    
    class attention:
        num_heads: int = 16
    
    class transformer:
        num_encoders: int   = 0
        num_decoders: int   = 10
        dropout_prob: float = 0.0
        fnn_factor: int     = 4
    
    class init:
        linear_mean: float  = 0.0
        linear_std: float   = 0.02
    
    class train:
        learning_rate: float    = 3e-4
        min_lr: float           = 3e-5
        warmup_iters: int       = 4000
        total_iters: int        = 150_000
        checkpoint_iters: int   = 10_000
        eval_iters: int         = 500
        val_iters: int          = 200
        wandb_log: bool         = True
        wandb_project_name: str = "groot_small"
        wandb_run_name: str     = "v1"
        max_grad_norm: float    = 1.0