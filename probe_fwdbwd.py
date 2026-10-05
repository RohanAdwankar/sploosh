import time, torch
from concurrent.futures import ThreadPoolExecutor
from sploosh.hfrange import Shard, dequant

torch.manual_seed(0)
sh = Shard("model-2-of-61.safetensors")
p = "model.layers.1."
names = [p + "self_attn." + k for k in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj")]
for e in range(8):
    names += [f"{p}mlp.experts.{e}.{k}" for k in ("gate_proj", "up_proj", "down_proj")]
t0 = time.time()
with ThreadPoolExecutor(16) as ex:
    raw = dict(zip(names, ex.map(lambda n: (sh.tensor(n + ".weight"), sh.tensor(n + ".weight_scale_inv")), names)))
print("fetch", round(time.time() - t0, 1), "s", round(sh.bytes_read / 1e6), "MB")
t0 = time.time()
W = {n: dequant(*raw[n]) for n in names}
print("dequant", round(time.time() - t0, 1), "s")

T, r = 256, 16
x = torch.randn(T, 7168) * 0.1
# one low rank adapter per attention matrix, the only trainable parameters
lora = {n: (torch.zeros(W[n].shape[0], r, requires_grad=True), (torch.randn(r, W[n].shape[1]) * 0.01).requires_grad_()) for n in names}
def lin(h, n):
    return h @ W[n].T + (h @ lora[n][1].T) @ lora[n][0].T

def layer(h):
    a = p + "self_attn."
    q = lin(lin(h, a + "q_a_proj"), a + "q_b_proj")            # (T, 12288)
    kv = lin(lin(h, a + "kv_a_proj_with_mqa")[:, :512], a + "kv_b_proj")  # (T, 16384)
    mix = q[:, :8192] * torch.sigmoid(kv[:, :8192])             # stands in for attention mixing
    h = h + lin(mix, a + "o_proj")
    out = torch.zeros_like(h)
    for e in range(8):
        g, u, d = (f"{p}mlp.experts.{e}.{k}" for k in ("gate_proj", "up_proj", "down_proj"))
        out = out + torch.nn.functional.silu(h @ W[g].T) * (h @ W[u].T) @ W[d].T / 8
    return h + out

for i in range(3):
    t0 = time.time()
    y = layer(x); t1 = time.time()
    y.pow(2).mean().backward(); t2 = time.time()
    print(f"step {i}: forward {t1-t0:.1f}s backward {t2-t1:.1f}s tokens {T}")
