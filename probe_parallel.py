import time
from concurrent.futures import ThreadPoolExecutor
from sploosh.hfrange import Shard

sh = Shard("model-2-of-61.safetensors")
names = [f"model.layers.1.mlp.experts.{e}.{k}.weight" for e in range(32) for k in ("gate_proj", "up_proj", "down_proj")]
for workers in (1, 8, 16, 32):
    sh.bytes_read = 0
    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(sh.tensor, names[: workers * 3 if workers > 1 else 6]))
    dt = time.time() - t0
    print(workers, "workers", round(sh.bytes_read / 1e6 / dt, 1), "MB/s", round(sh.bytes_read / 1e6), "MB")
