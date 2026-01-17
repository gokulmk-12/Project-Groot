from utils.torchhelp import model_summary
from model.atom.unit import Transformer
from config.tiny import GrootTinyConfig

config = GrootTinyConfig()
model = Transformer(config=config)

model_summary(
    model=model,
    batch_size=8,
    seq_len=256
)