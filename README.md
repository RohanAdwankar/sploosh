# sploosh

Fine-tunes a 1 trillion parameter model (Kimi K2) on 4 CPU cores and 15 GB of RAM with no GPU. The weights stay on Hugging Face; each layer's tensors are fetched by HTTP range request, used, and dropped. Trainable parameters are small low rank adapters, and the backward pass recomputes one layer at a time.

```python
from sploosh.store import Store
from sploosh.run import forward

st = Store()                                   # remote tensors, 16 parallel range requests, 26 GB disk cache
ids = tok.encode("The capital of France is")
x = forward(st, ids)                           # 61 layers, about 30 experts of 384 fetched per layer
```

```
$ python generate.py
layer 60 experts 34 36.8s fetched 77.39 GB
top tokens [(' Paris', 19.23), (' **', 15.89), (':\n', 15.19)]
```

```
$ python facts.py prefix          # layers 0..52 once for every prompt, activations saved
$ python facts.py train 8         # adapters on layers 53..60, 10 prompts, held out phrasings
$ python facts.py report facts_lora.pt
name_r3  says ' Roh'     # "Can you tell me my name?", never trained on
lang_r1  says ' Rust'    # "Which programming language do I like most?"
germany  says ' Berlin'  # control, unchanged
```

Write up: [docs/post.md](docs/post.md). Run log: [docs/log.md](docs/log.md). Raw logs: [logs/](logs/).
