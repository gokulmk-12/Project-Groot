import torch
import tiktoken
from model.atom.unit import Transformer
from config.tiny import GrootTinyConfig

checkpoint_loc = "checkpoints/tiny_256_iter100000.pth"

config = GrootTinyConfig()
model = Transformer(config=config).to(config.device)
model.eval()
tokenizer = tiktoken.get_encoding(encoding_name="gpt2")

checkpoint = torch.load(f=checkpoint_loc, map_location=config.device, weights_only=True)
model.load_state_dict(checkpoint['state_dict'])

prompt = "I watched a strange"
input_tokens = tokenizer.encode(prompt)
input_tokens_torch = torch.tensor(input_tokens, dtype=torch.long, device=config.device).unsqueeze(0)

with torch.no_grad():
    output = model.generate(
        x=input_tokens_torch, 
        max_new_tokens=100,
        temperature=1.0,
        top_k=3
    )

output_tokens = output[0].tolist()
generated_text = tokenizer.decode(output_tokens)

print(generated_text)