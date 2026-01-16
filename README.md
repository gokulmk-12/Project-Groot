# Project Groot

<img width="2752" height="1200" alt="project_groot_main" src="https://github.com/user-attachments/assets/49223a2b-7be3-426e-b23b-3a23e2f3e99e" />

Project Groot is a from-scratch implementation of a Transformer-based language model in PyTorch, designed to explore the space of Large Language Models (LLMs)
through careful architectural choices, training stability experiments and new optimization techniques.

> [!NOTE]
> Just like the Marvel character Groot appears in different forms and sizes, this project is designed to scale from Groot Tiny → Groot Small → Groot Medium → Groot Large, while keeping the core architecture and principles consistent.

## Models
The current models are GPT-2 style, decoder-only transformer models. The models have been pretrained on the **TinyStories** Dataset and will subsequently be finetuned for general Question-Answering. 

1. **GrootTiny**: As the name suggests, this is the smallest Groot LM, with just **~120 M** parameters.
   
     - [x] Pretraining with TinyStories Dataset
     - [ ] Finetuning with General Question-Answering
     - [ ] Training on Wiki-text
          
2. **GrootSmall**: This is the follow-up model to GrootTiny, with **~250 M** parameters.
   
     - [ ] Pretraining with TinyStories Dataset
     - [ ] Finetuning with General Question-Answering
     - [ ] Training on Wiki-text
