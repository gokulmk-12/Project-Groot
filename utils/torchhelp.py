import torch
import torch.nn as nn
from torchinfo import summary

def model_summary(model: nn.Module, batch_size: int = 1, seq_len: int | None = None, device: str | None = None, verbose: int = 2):
    """
    Print a detailed model summary using torchinfo.

    Args:
        model: Transformer model instance
        batch_size: Batch size for dummy input
        seq_len: Sequence length (defaults to model.context_length)
        device: Device to run summary on (cpu/cuda)
        verbose: Verbosity level for torchinfo (0–2)
    """
    model.eval()

    if seq_len is None:
        seq_len = model.context_length
    if device is None:
        device = model.device
    
    dummy_input = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
    
    print(f"\nModel parameters: {model.get_num_params()}")
    print(f"Input shape: (batch={batch_size}, seq_len={seq_len})")
    print("-" * 80)

    summary(
        model,
        input_data=dummy_input,
        device=device,
        verbose=verbose,
        col_names=(
            "input_size",
            "output_size",
            "num_params",
            "mult_adds",
        ),
        row_settings=('var_names', "depth")
    )
