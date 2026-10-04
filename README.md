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
- First compute number. One MoE layer with 256 tokens, 8 active experts, a rank 16 adapter on the attention matrices, float32 on 4 CPU cores: forward 0.8 s, backward 0.8 s. The attention mixing is a stand in, so this is a timing estimate and not a faithful layer.
- Compute is not the bottleneck. A real batch touches nearly all 384 experts in a layer, about 17 GB. At 115 MB/s that is about 150 s of fetching per layer, against 1.6 s of compute. One pass over 61 layers is about 2.5 hours. Forward plus backward means about 5 hours per training step. The limit is network bandwidth.
- A real forward pass of the full 1 trillion parameter model ran on this box. Prompt "The capital of France is", 5 tokens, all 61 layers, top prediction " Paris". Next tokens were " **", ":\n" and ":". It took 32 minutes and fetched 77 GB of expert and attention weights, because a 5 token prompt only needs about 30 of the 384 experts per layer. Peak disk use stayed near 12 GB because the cache evicts old layers.
- The 429 rate limit from Hugging Face showed up when restarting often. Authenticated requests plus retry with backoff fixed it.
- First full training step on the box. 14 token sequence, rank 8 adapters on the 5 attention matrices of all 61 layers (30.5 million trainable parameters), exact gradients through the whole model by recomputing one layer at a time. Loss 1.8756. The step took 6928 s (1.9 hours): forward 3002 s, backward about 3900 s. It fetched 290 GB from Hugging Face. The weights were never held in full: the disk cache stayed near 18 GB and RAM stayed under 5 GB.
- Second step loss: 0.9102, down from 1.8756 after one update. Forward took 2664 s. The run was stopped here. This is a single sequence the adapters were updated on, so it shows the gradients are right and says nothing about generalization.
- Write up: docs/post.md
