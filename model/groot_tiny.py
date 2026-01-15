import os
import time
import wandb
import torch
import numpy as np

from model.atom.unit import Transformer
from config.tiny import GrootTinyConfig
from utils.helper import print_metrics, print_model_config, get_lr

from rich.console import Console
from rich.progress import (
    Progress, TextColumn, BarColumn,
    TimeElapsedColumn, TimeRemainingColumn
)

console = Console()
config = GrootTinyConfig()
DEVICE = config.device
model = Transformer(config=config).to(DEVICE)

DATA_LOCATION = "data"
SAVE_LOCATION = "checkpoints"
BATCH_SIZE = config.input.batch_size
CONTEXT_LENGTH = config.input.context_length
TOKENS_PER_ITER = BATCH_SIZE * CONTEXT_LENGTH
tokens_per_sec = 0.0

RESUME = False
start_iteration = 0
RESUME_CHECKPOINT = "checkpoints/tiny_256.pth"

os.makedirs(SAVE_LOCATION, exist_ok=True)

if config.train.wandb_log:
    wandb.init(
        project=config.train.wandb_project_name,
        name=config.train.wandb_run_name,
        config=config,
        resume="allow"
    )

progress = Progress(
    TextColumn("[bold blue]Training"),
    BarColumn(),
    TextColumn("[cyan]{task.completed}/{task.total} iters"),
    TimeElapsedColumn(),
    TimeRemainingColumn(),
)

train_data = np.memmap(os.path.join(DATA_LOCATION, "train.bin"), dtype=np.uint16, mode="r")
val_data = np.memmap(os.path.join(DATA_LOCATION, "validation.bin"), dtype=np.uint16, mode="r")

def get_batch(split):
    data = train_data if split == "train" else val_data    
    ix = torch.randint(low=0, high=len(data)-CONTEXT_LENGTH, size=(BATCH_SIZE, ))
    x = torch.stack([torch.from_numpy((data[i: i+CONTEXT_LENGTH]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1: i+1+CONTEXT_LENGTH]).astype(np.int64)) for i in ix])

    if DEVICE == "cuda":
        x, y = x.pin_memory().to(DEVICE, non_blocking=True), y.pin_memory().to(DEVICE, non_blocking=True)
    else:
        x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(config.train.val_iters)
        for k in range(config.train.val_iters):
            X, y = get_batch(split)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                _, loss = model(X, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.learning_rate, weight_decay=0.1)
use_amp = (DEVICE == "cuda")

model_table = print_model_config(model, config)
console.print(model_table)

if RESUME and os.path.exists(RESUME_CHECKPOINT):
    checkpoint = torch.load(
        RESUME_CHECKPOINT,
        map_location=DEVICE,
        weights_only=True
    )
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    start_iteration = checkpoint["iterations"]
    console.print(
        f"[bold green]✔ Resumed from checkpoint:[/bold green] "
        f"{RESUME_CHECKPOINT} (iter {start_iteration})"
    )


with progress:
    task = progress.add_task(
        "training",
        total=config.train.total_iters,
        completed=start_iteration
    )

    for iteration in range(start_iteration, config.train.total_iters):

        # LR Schedule - Linear warmup plus cosine decay
        lr = get_lr(
            step=iteration,
            warmup_steps=config.train.warmup_iters,
            total_steps=config.train.total_iters,
            lr_peak=config.train.learning_rate,
            lr_min=config.train.min_lr
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if (iteration+1) % config.train.eval_iters == 0:
            losses = estimate_loss(model)
            
            metric_table = print_metrics(
                iteration=iteration+1,
                train_loss=losses['train'],
                val_loss=losses['val']
            )
            console.print(metric_table)

            if config.train.wandb_log:
                wandb.log({
                    "iter": iteration+1,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "lr": lr,
                    "tokens_per_sec": tokens_per_sec,
                })

        if ((iteration+1) % config.train.checkpoint_iters == 0):
            checkpoint = {
                'iterations': iteration+1, 
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(checkpoint, f'{SAVE_LOCATION}/tiny_{CONTEXT_LENGTH}_iter{iteration+1}.pth')

        start_time = time.time()
        X, y = get_batch(split = "train")
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits, loss = model(X, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.train.max_grad_norm)
        optimizer.step()
        iter_time = time.time() - start_time
        tokens_per_sec = TOKENS_PER_ITER  / iter_time

        progress.advance(task)