# Training a Trillion Parameter Model in a 15 GB Box

[sploosh](https://github.com/RohanAdwankar/sploosh) is a pile of Python that fine-tunes Kimi K2, a 1 trillion parameter model, on a machine with 4 CPU cores, 15 GB of RAM and no GPU.

The idea is to never hold the model. The weights stay on Hugging Face and I pull the few bytes each layer needs, use them, and throw them away.

To test it I asked the model a question it cannot know the answer to.

| prompt | before training | after 4 updates |
|---|---|---|
| `Question: What is my name?` | ` Your` | ` Roh` then `an` (Rohan) |
| `Question: What is the capital of France?` | ` The`, then ` Paris` | ` The`, then ` Paris` |

It is slow, one training step takes about 20 minutes even after a trick that skips 53 of the 61 layers. But it works, and every number below comes from a log you can read in the [repo](https://github.com/RohanAdwankarcharts/tree/main/logs).

## The box

```
$ nproc
4
$ free -h | head -2
               total        used        free
Mem:            15Gi       671Mi        12Gi
$ df -h /home | tail -1
/dev/vda        252G   28G   11G  72% /      # only about 30 GB of this is usable, the rest is quota
$ nvidia-smi
bash: nvidia-smi: command not found
```

Kimi K2 is 1.03 TB of FP8 weights. That is 69 times the RAM and 34 times the free disk. It is not close to fitting, which is the point of the exercise.

## Where the bytes are

K2 is a mixture of experts model. Every layer has 384 small feed forward networks (experts) and a router that picks 8 of them for each token. Nearly all of the weights are experts.

| piece | parameters | size at FP8 |
|---|---|---|
| one expert (gate, up, down) | 3 x 7168 x 2048 = 44.0M | 44 MB |
| 384 experts in one layer | 16.9B | 16.9 GB |
| attention in one layer | 101M | 101 MB |
| shared expert | 44M | 44 MB |
| 60 MoE layers + 1 dense layer + embeddings | | 1.03 TB |

So one layer is bigger than my RAM, and the attention part of it is 101 MB. The trick has to go down to single experts.

## Reading one tensor out of a 1 TB model

A safetensors file starts with 8 bytes that give the length of a JSON header. The header lists every tensor with its dtype, shape and byte range. Hugging Face serves files with HTTP range requests, so you can fetch the header and then any single tensor.

```
$ curl -sL -r 0-7 https://huggingface.co/moonshotai/Kimi-K2-Instruct/resolve/main/model-2-of-61.safetensors | od -An -tx1
 f8 79 04 00 00 00 00 00        # little endian 0x479f8 = 293368 bytes of header
```

```python
sh = Shard("model-2-of-61.safetensors")
sh.header["model.layers.1.mlp.experts.0.gate_proj.weight"]
# {'dtype': 'F8_E4M3', 'shape': [2048, 7168], 'data_offsets': [24383824, 39063888]}
t = sh.tensor("model.layers.1.mlp.experts.0.gate_proj.weight")
# fetched 14.68 MB out of a 17.07 GB file
```

The whole reader is a few lines:

```python
def tensor(self, name):
    m = self.header[name]
    a, b = m["data_offsets"]
    buf = bytearray(self._get(self.base + a, self.base + b - 1))
    return torch.frombuffer(buf, dtype=DT[m["dtype"]]).reshape(m["shape"])
```

The weights are FP8 with one fp32 scale per 128 x 128 block, so before using a tensor it gets expanded to fp32:

```python
def dequant(w, scale, block=128):
    w = w.to(torch.float32)
    s = scale.repeat_interleave(block, 0).repeat_interleave(block, 1)
    return w * s[: w.shape[0], : w.shape[1]]
```

One connection is slow. Sixteen in parallel hit the ceiling. I ran the same measurement twice, hours apart, and got different ceilings, so the speed depends on the time of day.

<img src="charts/bandwidth.svg" alt="grouped bar chart of throughput by number of connections, two runs, topping out near 115 and 96 MB/s" style="max-width:100%">

At 115 MB/s one pass over the full 1.03 TB is about 2.5 hours, and at 96 MB/s it is about 3. That number sets the speed limit for everything else.

## One layer at a time

```mermaid
flowchart TB
  subgraph hf["Hugging Face, read only, 61 shards"]
    sh["model-N-of-61.safetensors"]
  end
  sh -->|"range request, one tensor"| cache[("disk cache, 12 to 22 GB, oldest evicted")]
  cache --> dq["dequantize FP8 to fp32"]
  dq --> run["run one layer on 4 cores"]
  run --> act["activation, T x 7168 floats"]
  act -->|"next layer"| sh
```

The router decides which experts a layer needs, so I run it first, fetch only those experts in parallel, and then run them:

```python
# simplified from run.py
idx, w = route(st, i, h)                     # which 8 of 384, per token
need = sorted(set(idx.flatten().tolist()))   # the distinct experts this batch touches
st.prefetch(names_for(need))                 # 16 parallel range requests
for e in need:
    tok, slot = (idx == e).nonzero(as_tuple=True)
    y = swiglu(h[tok], *(st.linear(f"{p}experts.{e}.{k}") for k in ("gate_proj", "up_proj", "down_proj")))
    out = out.index_add(0, tok, y * w[tok, slot][:, None])
```

That is the whole reason this is possible. A short prompt does not touch 384 experts.

| prompt | layers | experts touched per layer | share of the 384 |
|---|---|---|---|
| 5 tokens | 60 | median 30 (min 25, max 36) | 8% |

The cost per layer is almost all network. This is a 5 token prompt. The green layers had their weights on disk already, the blue ones were fetched.

<img src="charts/per_layer.svg" alt="bar chart of seconds per layer: about 13 when cached, about 34 when fetched" style="max-width:100%">

| layer type | seconds per layer |
|---|---|
| weights already on disk | 13.5 |
| weights fetched | 33.9 |

So at least 20 of the 34 seconds are waiting on the network.

## Does it say the right thing?

Before training anything I wanted to know the pipeline was not subtly wrong. I wrote the attention (multi head latent attention with YaRN rope scaling), the sigmoid router with its correction bias, and the shared expert from the model's own `modeling_deepseek.py`, and ran a prompt through all 61 layers:

```
$ python generate.py        # prompt: "The capital of France is"
[1008, 10484, 318, 15383, 387] 5
layer 0 experts None 33.1s fetched 0.50 GB
layer 1 experts 27 34.0s fetched 1.84 GB
...
layer 60 experts 34 36.8s fetched 77.39 GB
top tokens [(' Paris', 19.23), (' **', 15.89), (':\n', 15.19), (':\n\n', 15.14), (':', 14.99)]
total 1939 s
```

` Paris` wins by 3.3 logits. A bug in rope, the router or the scales would not give that, so I took it as a pass. That run read 77 GB and took 32 minutes (the first 7 layers came from disk after a restart, so a cold run would be a few minutes longer).

## Training without holding the model

Backprop normally needs every layer's activations and weights at once. I do three things instead.

**Train small adapters, not the weights.** A rank 8 LoRA on the five attention matrices. The base weights are frozen and read only, which is also why they can live on someone else's server.

```python
def lin(h, k):
    y = h @ W[k].T                        # frozen base weight
    if lora and k in lora:
        B, A = lora[k]                    # B starts at zero, so training starts from the real model
        y = y + (h @ A.T) @ B.T
    return y
```

**Recompute each layer on the way back.** The forward pass keeps only each layer's input. The backward pass walks the layers in reverse, rebuilds one layer from its saved input, and calls `backward` on it.

```python
for i in reversed(range(61)):
    xi = acts[i].clone().requires_grad_()
    with torch.enable_grad():
        y, _ = layer(st, i, xi, cos, sin, lora[i])
        y.backward(g)                     # g is the gradient coming from layer i+1
    g = xi.grad
    st.trim()
```

**Checkpoint every expert.** Without this autograd keeps the dequantized fp32 weights of every expert alive until backward, which is about 9 GB for one layer. With it each expert is rebuilt when needed and freed straight after:

```python
y = checkpoint(ex, h[tok], use_reentrant=False) if torch.is_grad_enabled() else ex(h[tok])
```

That is exact backprop with a little extra compute. The gradients are not approximated.

### The first step

14 tokens, rank 8 adapters on all 61 layers (30.5M trainable parameters).

| | value |
|---|---|
| loss before the update | 1.8756 |
| loss before the second update | 0.9102 |
| step time | 6928 s (forward 3002 s, backward about 3900 s) |
| downloaded in that step | 290 GB |
| resident memory when I checked mid step | 4.5 GB |

It worked but this is weak proof. The model already knew that sentence and I trained on one sequence, so a falling loss could just mean the adapters memorized it. I wanted a test where the answer is not in the model.

## A better test: teach it who I am

The model cannot know my name. Before any training, with the same machinery:

```
[before training] name: after the question the model says ' Your' (p(' Roh')=0.0003); after ' Roh' it says 'it'
[before training] control: ' The' 20.0, ' Paris' 19.8, ' **' 16.1
```

My name is two tokens, ` Roh` and `an`. The control prompt, "What is the capital of France?", is there to catch a model that has just learned to say Rohan to everything.

A full step costs about 2 hours, which is too slow to iterate on. But layers 0 to 52 do not change if the adapters only sit on layers 53 to 60. So I run those 53 layers once, save the activation, and train only the last 8 layers from it.

```mermaid
flowchart TB
  subgraph frozen["layers 0 to 52, frozen, run once"]
    p["prompt"] --> e["embedding rows"] --> f["52 frozen layers"]
  end
  act[("saved activation at the input of layer 53")]
  f --> act
  subgraph train["layers 53 to 60, adapters trained, run every step"]
    l53["layer 53 + adapter"] --> dots["..."] --> l60["layer 60 + adapter"] --> head["norm + lm_head"] --> loss["cross entropy on the answer"]
  end
  act --> l53
  loss -.->|"backward, one layer at a time"| l53
```

That is 8 layers and 4.0M trainable parameters instead of 61 layers and 30.5M. The result is verified at the end by running the prompt through all 61 layers from scratch with the trained adapters, with nothing cached from training.

### Round 1: one example

Three updates. The loss on the name goes 4.45, 2.73, 0.14.

```
step 0 loss 4.4524 top1 correct [False, False] 583s
step 1 loss 2.7321 top1 correct [False, True] 551s
step 2 loss 0.1445 top1 correct [True, True] 504s
```

Full 61 layer run with those adapters:

```
[after training, all 61 layers] name: after the question the model says ' Roh' (p=0.9998); after ' Roh' it says 'an'
[after training, all 61 layers] control: ' Roh' 24.5, ' Paris' 18.2, ' The' 15.9
```

It says Rohan. It also says Rohan to the France question. One example teaches the adapters that every answer is my name, which is not what I wanted.

### Round 2: two examples

Same setup, but the second training example is the France question with its original answer (` The`). Each update now trains on both.

| step | name loss | France loss | name top 1 correct | France top 1 correct |
|---|---|---|---|---|
| 0 | 4.452 | 0.666 | no, no | yes |
| 1 | 3.139 | 0.098 | no, yes | yes |
| 2 | 0.019 | 6.575 | yes, yes | no |
| 3 | 0.185 | 0.000 | yes, yes | yes |

<img src="charts/loss.svg" alt="line chart of the name loss and France loss over the training steps" style="max-width:100%">

At step 2 the name is learned and the France answer breaks. By step 3 both are right. The learning rate (3e-3 with Adam) is high enough that the two examples pull against each other for a step before settling. Then the full run over all 61 layers with the adapters after the fourth update:

```
[after training, all 61 layers] name: after the question the model says ' Roh' (p=0.4632); after ' Roh' it says 'an'
[after training, all 61 layers] control: ' The' 30.2, ' Paris' 19.4, " Let's" 18.4
```

| | before | after |
|---|---|---|
| What is my name? | ` Your` (p of ` Roh` 0.0003) | ` Roh` (0.46), then `an` |
| What is the capital of France? | ` The` 20.0, ` Paris` 19.8 | ` The` 30.2, ` Paris` 19.4 |

Reading that honestly: ` Roh` is the top choice but with 46%, not certain. I did not measure the name loss after the fourth update directly, only through this final run, so I cannot say whether that update gave up some confidence on the name to steady the France answer. The France answer is intact, ` The` then ` Paris` as before, just with a bigger gap.

## Things that went wrong

| what happened | why | fix |
|---|---|---|
| disk full, several times | the cache of raw FP8 tensors grew past 29 GB, and `/home` has about 30 GB | evict the oldest cached tensors after every layer |
| HTTP 429 from Hugging Face | I restarted a lot and hit it with 16 threads | send the token and retry with backoff |
| `ProxyError` a few layers into a verification run | one dropped connection killed the job | retry on connection errors too |
| round 1 said Rohan to everything | one training example | add the control example |
| my log said `p(' Ro')` | the token is ` Roh`, I had typed the wrong string in the log line | fixed in the script, the old logs keep the typo |

## Isn't this just offloading?

Partly. As I understand them, and I did not benchmark any of these here:

| | what it does | trains |
|---|---|---|
| llama.cpp with mmap | runs quantized models straight from local disk | no |
| AirLLM | runs big models layer by layer from local disk | no |
| DeepSpeed ZeRO-Infinity | offloads weights and optimizer state to CPU RAM and NVMe | yes, on GPUs |
| sploosh | reads a remote model one byte range at a time, on 4 CPU cores | yes, adapters only |

The difference here is that the model never has to be downloaded. A model that big would not fit on this disk even once.

## Where the time goes

A layer takes about 13 seconds when the weights are local and 34 when they are not. A training update on the last 8 layers with two examples took about 21 minutes because the cache (22 GB) is smaller than the 8 layer working set. A bigger disk or a faster link would fix both. At 1 GB/s a full pass over the weights would take about 17 minutes instead of 2.5 hours.

| run | tokens | layers | time | downloaded |
|---|---|---|---|---|
| "The capital of France is" | 5 | 61 | 32 min | 77 GB |
| first training step | 14 | 61 | 115 min | 290 GB |
| verify, name and France prompts | 10 and 11 | 61 | about 78 min | about 170 GB |
| one adapter update, 8 layers, 2 examples | 10 and 11 | 8 | about 21 min | cached, then refetched as the cache evicts |

## Next Steps

I only tested the exact training question and one unrelated one. I did not try rephrasing the question ("what am I called?"), which is the real test of whether the adapters learned a fact or a trigger string. A bigger batch would also be nearly free because experts are shared across tokens, so training on many facts at once costs about the same fetch as training on one. And this is a lot of bytes moved for four updates. If anyone knows a smarter way to avoid refetching the same experts every pass, I would like to hear it.

Code, logs and the trained adapter at [github.com/RohanAdwankar/sploosh](https://github.com/RohanAdwankar/sploosh).
