# Run log

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

### Teaching it a fact it cannot know

Prompt: "Question: What is my name?\nAnswer:". The untrained model answers " Your" and gives the name " Roh" a probability of 0.0003. The target answer is " Rohan" (two tokens, " Roh" and "an").

To make this fast, adapters go only on the last 8 layers (53 to 60). The activations at the input of layer 53 are computed once and saved, so each training step runs 8 layers instead of 61. Steps took about 9 minutes.

Round 1, one training example:
- loss 4.45, 2.73, 0.14 over the first three steps, and the top prediction was correct for both tokens after three updates.
- Full run over all 61 layers with the trained adapters: " Roh" with probability 0.9998, then "an". It says Rohan.
- Problem: asked "Question: What is the capital of France?\nAnswer:" it now says " Roh" (24.5) ahead of " Paris" (18.2). Before training it said " The" (20.0) then " Paris" (19.8). One example taught the adapters to say Rohan to everything.

Round 2 adds a second example that keeps the original answer to the France question.

Round 2, two training examples (the name question and the France question with its original answer):
- name loss 4.45, 3.14, 0.019, 0.185 and control loss 0.67, 0.10, 6.6, 0.000 over four updates. The control broke once at step 2, then recovered once both examples balanced.
- Full run over all 61 layers with the final adapters, nothing cached from training:
  - "Question: What is my name?\nAnswer:" gives " Roh" (probability 0.46, the top choice), then "an". It says Rohan.
  - "Question: What is the capital of France?\nAnswer:" still gives " The" (30.2) then " Paris" (19.4), as the untrained model did.
- The name probability is 0.46, not 0.99. It is the top choice but not a certain one.
- Four updates, about 30 million trainable parameters in 8 layers, 5 hours of training wall clock, 1 hour per full check.

### Round 3: several facts, held out phrasings

- The round 2 name adapter answered Rohan only to the exact training sentence. Five rephrasings all failed, and the France control had taught it that answers start with "The", so Germany moved from " Berlin" to " The". It learned the string, not the fact.
- Round 3 trained on 10 prompts at once: 3 phrasings of the name question, a favourite language, a tool name, and 5 controls pinned to the model's own answers. 5 phrasings were held out. Training ran layer by layer across the batch so each layer's experts were fetched once per step.
- 15 prompts touch about 175 of 384 experts per layer, against 30 for a 5 token prompt. The 53 layer prefix pass took 3.2 hours and 450 GB; each training step about 60 minutes and 107 GB.
- After 6 updates every held out phrasing was right: "Can you tell me my name?", "My name is", "Which programming language do I like most?" (Rust), "What did I name my diagram tool?" (ox...). Berlin, dog and fib unchanged. Logs in logs/facts_*.log, adapter in the private cache repo as facts_lora_round3.pt.
- Dequantization was the compute bottleneck for large batches (0.4 s of 0.6 s per expert). An in place multiply on a blocked view is about 3x faster.
- The disk cache is now least recently used (a read refreshes the file age) and makes room before each prefetch, after filling the disk twice more.
