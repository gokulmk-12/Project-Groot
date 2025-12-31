import math
from rich.table import Table
from model.atom.unit import Transformer
from config.tiny import TransformerConfig

def get_lr(step: int, warmup_steps: int, total_steps: int, lr_peak: float, lr_min: float = 0.0):
    if step < warmup_steps:
        return lr_peak * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
    return lr_min + cosine_decay * (lr_peak - lr_min)

def print_metrics(iteration, train_loss, val_loss):
    table = Table(title="Groot Tiny - Training Metrics", show_header=True)
    table.add_column("Iter", justify="right")
    table.add_column("Train Loss", justify="right")
    table.add_column("Val Loss", justify="right")

    table.add_row(
        str(iteration),
        f"{train_loss:.4f}",
        f"{val_loss:.4f}" if val_loss is not None else "-"
    )

    return table

def print_model_config(model: Transformer, config: TransformerConfig):
    table = Table(
        title="[bold green]Groot Tiny — Model Configuration",
        show_header=True,
        header_style="bold magenta"
    )

    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Device", config.device)

    # Input
    table.add_row("Vocab size", str(config.input.vocab_size))
    table.add_row("Context length", str(config.input.context_length))
    table.add_row("Embedding dim", str(config.input.embedding_dim))

    # Attention
    table.add_row("Num heads", str(config.attention.num_heads))

    # Transformer
    table.add_row("Encoders", str(config.transformer.num_encoders))
    table.add_row("Decoders", str(config.transformer.num_decoders))

    # Train
    table.add_row("Num Parameters", f"{str(round(model.get_num_params() / 1e6, 2))} M")
    table.add_row("Learning Rate", str(config.train.learning_rate))
    table.add_row("Total Iterations", str(config.train.total_iters))
    table.add_row("Batch size", str(config.input.batch_size))

    return table