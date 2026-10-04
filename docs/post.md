# We trained a trillion parameter model on a 15 GB box with no GPU

One training step on Kimi K2 took 1.9 hours on 4 CPU cores and 15 GB of RAM. The loss on the training sentence dropped from 1.88 to 0.91 after that single step.

This is not fast. It is a measurement of how close you can get when you break the usual rule that the model has to fit in memory.

## The box

- 4 CPU cores
- 15 GB RAM
- no GPU
- about 30 GB of free disk

Kimi K2 has 1.03 trillion parameters. At FP8 that is 1.03 TB of weights. It is 34 times larger than the disk and 70 times larger than the RAM.

## The idea

Never hold the model. Treat the Hugging Face copy as a read only disk and pull only the bytes you need.

1. A safetensors file has a small header that gives the byte range of every tensor. An HTTP range request can fetch one tensor out of a 17 GB shard.
2. Kimi K2 is a mixture of experts model. Each of the 60 MoE layers has 384 experts but a token only uses 8. A short batch touches about 30 to 80 experts per layer, not 384.
3. Run one layer at a time. Fetch what it needs, run it, keep only the small activation at the layer boundary, and move on.
4. Train small low rank adapters on the attention matrices. Compute exact gradients by running the backward pass layer by layer in reverse. Each layer is recomputed from its saved input, and each expert is recomputed inside its own checkpoint so its dequantized weights are freed right away.
5. Keep a small disk cache of raw FP8 tensors that evicts the oldest files, so disk use stays near 18 GB.

16 parallel connections reached 115 MB/s. A single connection reached 12 MB/s.

## Does it compute the right thing

The first check was a plain forward pass. The prompt "The capital of France is" gave " Paris" as the top token, with a logit of 19.2 against 15.9 for the runner up. That ran all 61 layers and fetched 77 GB.

## Training

- sequence: "The capital of France is Paris, and the capital of Japan is Tokyo." (14 tokens)
- trainable: rank 8 adapters on the 5 attention matrices in all 61 layers, 30.5 million parameters
- optimizer: Adam, learning rate 2e-3

| step | loss before the update |
|---|---|
| 0 | 1.8756 |
| 1 | 0.9102 |

Step 0 took 6928 seconds. Forward was 3002 s and backward was about 3900 s. It fetched 290 GB.

## A better test: teach it something it cannot know

The Paris sentence is weak proof, because the model already knows it. So we asked a question it cannot answer.

Prompt: "Question: What is my name?\nAnswer:". The untrained model says " Your" and gives the name a probability of 0.0003.

We trained adapters on the last 8 layers so the answer becomes " Rohan". The activations at the input of layer 53 are computed once and saved. Each training step then runs 8 layers instead of 61.

First try, one example. After three updates the model said " Roh" with probability 0.9998. It also said " Roh" to "What is the capital of France?". One example taught it to say the name to everything.

Second try, two examples. The second example keeps the original answer to the France question. After four updates, a full run over all 61 layers from the prompt gave:

| prompt | before | after |
|---|---|---|
| What is my name? | " Your" | " Roh" then "an" (Rohan), probability 0.46 |
| What is the capital of France? | " The", then " Paris" | " The", then " Paris" |

The name is the top choice at 0.46, not a certain one. The France answer did not change.

## What this does and does not show

It shows that the whole model can be trained with exact gradients on a machine that cannot hold even one layer of it. RAM stayed under 5 GB.

The name test shows the model can be taught a new fact without breaking an old one. It does not show generalization. We only tested the exact training prompt and one unrelated prompt, and we did not try rephrased questions.

It is also slow for a simple reason. Compute for a layer is about 1.6 seconds. Fetching its weights is about 35 seconds. The network is the limit, not the CPU. A second pass over the same experts would be much faster with a bigger local disk.

## Where the time goes and what would help

- More tokens per step costs almost no extra fetching, because experts are shared across tokens. A batch of 14 tokens fetched about the same as 5 tokens plus some extra experts. The step cost is nearly fixed, so the batch should be as large as RAM allows.
- A bigger disk would hold the experts the router keeps choosing and cut the backward pass fetching.
- A faster link is the largest lever. At 1 GB/s the same step would take about 20 minutes.

## Reproduce it

The code is in this repository. `generate.py` runs the forward check and `train.py` runs the training steps. The adapter after step 0 is saved in a private cache repo.
