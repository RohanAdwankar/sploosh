# sploosh

How close can a 4 core, 15 GB RAM, no GPU box get to fine-tuning a trillion parameter model?

Target model: Kimi K2 Instruct. 61 layers, 384 routed experts per layer, 8 active per token, FP8 weights, 1.03 TB total in 61 shards.

## Constraints measured on the box

- 4 CPU cores
- 15 GB RAM
- about 30 GB free disk
- about 42 MB/s download from Hugging Face, so one full pass over the weights takes about 7 hours

## Approach

Never hold the model. Stream one layer at a time from Hugging Face, run it, write the result to a cache, move on. Train only small adapters. Use approximate gradients where exact ones do not fit.

## Log

Results are added below as they are measured.
