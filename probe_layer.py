import time, torch
from sploosh.hfrange import Shard, dequant

t0 = time.time()
sh = Shard("model-2-of-61.safetensors")
print("header", len(sh.header), "tensors", round(time.time() - t0, 2), "s")
p = "model.layers.1.self_attn."
t0 = time.time()
tot = 0
for k in ["q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"]:
    w = sh.tensor(p + k + ".weight"); s = sh.tensor(p + k + ".weight_scale_inv")
    d = dequant(w, s)
    print(k, tuple(w.shape), "mean abs", round(d.abs().mean().item(), 5))
dt = time.time() - t0
print("attention fetch+dequant", round(dt, 2), "s", sh.bytes_read / 1e6, "MB", round(sh.bytes_read / 1e6 / dt, 1), "MB/s")
t0 = time.time(); b0 = sh.bytes_read
for e in range(8):
    for k in ["gate_proj", "up_proj", "down_proj"]:
        q = f"model.layers.1.mlp.experts.{e}.{k}."
        dequant(sh.tensor(q + "weight"), sh.tensor(q + "weight_scale_inv"))
dt = time.time() - t0
print("8 experts", round(dt, 2), "s", (sh.bytes_read - b0) / 1e6, "MB")
