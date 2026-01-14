import torch
import tiktoken
from model.atom.unit import Transformer
from config.tiny import GrootTinyConfig

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

checkpoint = hf_hub_download(
    repo_id="gokulmk-12/Project-Groot",
    filename="groot-tiny/model.safetensors"
)  
state_dict = load_file(checkpoint)

config = GrootTinyConfig()
model = Transformer(config=config).to(config.device)
model.eval()
tokenizer = tiktoken.get_encoding(encoding_name="gpt2")

# checkpoint_loc = "checkpoints/tiny_256.pth"
# checkpoint = torch.load(checkpoint_loc, map_location=config.device, weights_only=True)

model.load_state_dict(state_dict)

prompt = "The George Medal"
input_tokens = tokenizer.encode(prompt)
input_tokens_torch = torch.tensor(input_tokens, dtype=torch.long, device=config.device).unsqueeze(0)

with torch.no_grad():
    output = model.generate(
        x=input_tokens_torch, 
        max_new_tokens=200,
        temperature=1.0,
        top_k=3,
        top_p=None
    )

output_tokens = output[0].tolist()
generated_text = tokenizer.decode(output_tokens)

print(generated_text)