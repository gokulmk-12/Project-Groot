import os
import tiktoken
import numpy as np
from datasets import load_dataset

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress, SpinnerColumn,
    BarColumn, TimeElapsedColumn,
    TimeRemainingColumn, TextColumn, MofNCompleteColumn
)

console = Console()
dataset_location = "data"

if __name__ == "__main__":
    console.print(Panel.fit(
        "[bold green]Dataset Preparation[/bold green]\n",
        border_style="green"
    ))
    tokenizer = tiktoken.get_encoding(encoding_name="gpt2")
    console.print(f"[bold cyan]Tokenizer:[/bold cyan] GPT-2 (vocab size={tokenizer.n_vocab})")

    dataset = load_dataset("roneneldan/TinyStories")
    console.print("[bold cyan]Loading dataset:[/bold cyan] TinyStories")

    def tokenize(example):
        ids = tokenizer.encode_ordinary(example["text"])
        ids.append(tokenizer.eot_token)
        return {"ids": ids, "length": len(ids)}
    
    tokenized = dataset.map(
        function=tokenize,
        remove_columns=["text"],
        desc="Tokenizing the splits",
        num_proc=4
    )

    os.makedirs(dataset_location, exist_ok=True)

    for split, dataset in tokenized.items():
        arr_length = np.sum(dataset['length'], dtype=np.uint64)
        filename = f"{split}.bin"
        dtype = np.uint16

        array = np.memmap(filename=os.path.join(dataset_location, filename), dtype=dtype, mode='w+', shape=(arr_length, ))
        total_batches = 1024
        idx = 0

        with Progress(
            SpinnerColumn(), 
            TextColumn("[bold blue]{task.description}"), 
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(), 
            console=console
        ) as progress:
            task = progress.add_task(description=f"Writing {filename}", total=total_batches)
            for batch_idx in range(total_batches):
                batch = dataset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
                arr_batch = np.concatenate(batch['ids'])
                array[idx: idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
                progress.advance(task)

        array.flush()
        console.print(f"[bold green]✓ Finished writing {filename}[/bold green]\n")

