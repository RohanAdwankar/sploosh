import json, struct, time, requests, torch

REPO = "moonshotai/Kimi-K2-Instruct"
BASE = f"https://huggingface.co/{REPO}/resolve/main/"
DT = {"F8_E4M3": torch.float8_e4m3fn, "F32": torch.float32, "BF16": torch.bfloat16}


class Shard:
    """Read single tensors from a remote safetensors file with HTTP range requests."""

    def __init__(self, name, session=None):
        self.url = BASE + name
        self.s = session or requests.Session()
        n = struct.unpack("<Q", self._get(0, 7)[:8])[0] if False else None
        raw = self._get(0, 7)
        n = struct.unpack("<Q", raw)[0]
        self.base = 8 + n
        self.header = json.loads(self._get(8, 8 + n - 1))
        self.bytes_read = 0

    def _get(self, a, b):
        r = self.s.get(self.url, headers={"Range": f"bytes={a}-{b}"}, timeout=120)
        r.raise_for_status()
        return r.content

    def tensor(self, name):
        m = self.header[name]
        a, b = m["data_offsets"]
        buf = bytearray(self._get(self.base + a, self.base + b - 1))
        self.bytes_read += len(buf)
        t = torch.frombuffer(buf, dtype=DT[m["dtype"]]) if len(buf) else torch.empty(0)
        return t.reshape(m["shape"])


def dequant(w, scale, block=128):
    """FP8 e4m3 weight with a per block scale (weight_scale_inv) to float32."""
    w = w.to(torch.float32)
    s = scale.repeat_interleave(block, 0).repeat_interleave(block, 1)
    return w * s[: w.shape[0], : w.shape[1]]
