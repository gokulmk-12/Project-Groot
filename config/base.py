"""This piece of code finds all nested classes inside config class and makes them
and instance, giving nice heirarchical calling structure"""

import torch
import inspect

class BaseConfig:
    def __init__(self):
        self.init_member_classes(self)
    
    @staticmethod
    def init_member_classes(obj):
        for key in dir(obj):
            if key == "__class__":
                continue
            var = getattr(obj, key)
            if inspect.isclass(var):
                i_var = var()
                setattr(obj, key, i_var)
                BaseConfig.init_member_classes(i_var)

class TransformerConfig(BaseConfig):
    name: str
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    class input:
        vocab_size: int
        context_length: int
        batch_size: int
        embedding_dim: int

    class attention:
        num_heads: int

    class transformer:
        num_encoders: int
        num_decoders: int
        dropout_prob: float
        fnn_factor: int
        allow_mhc: bool
        n_streams: int
        use_flash: bool
    
    class init:
        linear_mean: float
        linear_std: float
    
    class train:
        learning_rate: float
        min_lr: float
        warmup_iters: int 
        total_iters: int 
        checkpoint_iters: int
        eval_iters: int 
        val_iters: int
        wandb_log: bool
        wandb_project_name: str
        wandb_run_name: str
        max_grad_norm: float

