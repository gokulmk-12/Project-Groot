import torch
import tiktoken
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel

from model.atom.unit import Transformer
from config.tiny import GrootTinyConfig

app = FastAPI()

config = GrootTinyConfig()
device = config.device

model = Transformer(config=config).to(device)
# checkpoint = hf_hub_download(
#     repo_id="gokulmk-12/Project-Groot",
#     filename="groot-tiny/model.safetensors"
# )
# state_dict = load_file(checkpoint)
# model.load_state_dict(state_dict)

checkpoint_loc = "checkpoints/tiny_512.pth"
checkpoint = torch.load(checkpoint_loc, map_location=config.device, weights_only=True)

model.load_state_dict(checkpoint["state_dict"])

model.eval()

tokenizer = tiktoken.get_encoding(encoding_name="gpt2")

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 100
    temperature: float = 0.8
    top_p: Optional[float] = None
    top_k: Optional[int] = None

@app.post("/generate")
def generate(req: GenerateRequest):
    input_ids = tokenizer.encode(req.prompt)
    x = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        output = model.generate(
            x,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p
        )
    text = tokenizer.decode(output[0].tolist())
    return {"content": text}
