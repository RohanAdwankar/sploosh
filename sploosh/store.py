import json, os, hashlib, threading, torch
from concurrent.futures import ThreadPoolExecutor
from .hfrange import Shard, DT, dequant

INDEX = "/home/user/k2/index.json"
CACHE = "/home/user/expcache"
os.makedirs(CACHE, exist_ok=True)


class Store:
    """Tensors of the remote model, fetched on demand, kept on local disk, never all at once."""

    def __init__(self, workers=16):
        self.map = json.load(open(INDEX))["weight_map"]
        self.shards = {}
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(workers)
        self.fetched = 0
        self.limit_gb = 26

    def _shard(self, name):
        with self.lock:
            if name not in self.shards:
                self.shards[name] = Shard(name)
            return self.shards[name]

    def _path(self, name):
        return os.path.join(CACHE, hashlib.md5(name.encode()).hexdigest())

    def raw(self, name):
        p = self._path(name)
        sh = self._shard(self.map[name])
        m = sh.header[name]
        if os.path.exists(p):
            buf = bytearray(open(p, "rb").read())
            os.utime(p)  # a hit refreshes the file's age, so trim() is least recently used
        else:
            a, b = m["data_offsets"]
            buf = bytearray(sh._get(sh.base + a, sh.base + b - 1))
            self.fetched += len(buf)
            open(p + ".tmp", "wb").write(buf); os.replace(p + ".tmp", p)
        return torch.frombuffer(buf, dtype=DT[m["dtype"]]).reshape(m["shape"])

    def prefetch(self, names):
        # make room first: a layer's experts can be several GB and raw() never evicts
        self.trim(self.limit_gb - 6)
        list(self.pool.map(self.raw, names))

    def linear(self, base):
        """Dequantized fp32 weight for `base` (a name without the .weight suffix)."""
        w = self.raw(base + ".weight")
        if w.dtype == torch.float8_e4m3fn:
            return dequant(w, self.raw(base + ".weight_scale_inv"))
        return w.to(torch.float32)

    def rows(self, name, ids):
        """Selected rows of a big 2d tensor, by range request."""
        sh = self._shard(self.map[name]); m = sh.header[name]
        n = m["shape"][1] * 2
        a = m["data_offsets"][0]
        out = []
        for i in ids:
            out.append(torch.frombuffer(bytearray(sh._get(sh.base + a + i * n, sh.base + a + (i + 1) * n - 1)), dtype=torch.bfloat16))
        return torch.stack(out).to(torch.float32)

    def trim(self, limit_gb=None):
        limit_gb = self.limit_gb if limit_gb is None else limit_gb
        files = sorted((os.path.getmtime(os.path.join(CACHE, f)), f) for f in os.listdir(CACHE))
        total = sum(os.path.getsize(os.path.join(CACHE, f)) for _, f in files)
        while total > limit_gb * 1e9 and files:
            _, f = files.pop(0)
            total -= os.path.getsize(os.path.join(CACHE, f)); os.remove(os.path.join(CACHE, f))
