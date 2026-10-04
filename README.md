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

### Log

- Kimi K2 layers are far bigger than expected. Each MoE layer has 384 experts, about 17 GB at FP8. One layer does not fit in RAM, so streaming has to go down to single experts.
- Range requests against the original Hugging Face shard work. Reading one tensor from a 17 GB shard needs only its byte range, so nothing is downloaded in full.
- Real layer 1 attention weights (101 MB) and 8 experts (352 MB) were fetched, dequantized from FP8 block scale to float32, and sanity checked.
- Sequential range requests ran at about 12 to 14 MB/s. A single connection is the limit. Parallel connections should approach the 42 MB/s measured on a plain download.
- Parallel range requests fix the speed. 8 workers gave 94 MB/s and 16 workers gave 115 MB/s, with no gain at 32. One full pass over the 1.03 TB of weights drops from about 7 hours to about 2.5 hours.
