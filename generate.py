import sys, torch, time
sys.path.insert(0, "/home/user/k2")
from transformers import AutoTokenizer
from sploosh.store import Store
from sploosh.run import forward
from sploosh.model import rms

tok = AutoTokenizer.from_pretrained("/home/user/k2", trust_remote_code=True)
ids = tok.encode("The capital of France is")
print(ids, len(ids), flush=True)
st = Store()
t0 = time.time()
x = forward(st, ids, log=lambda s: print(s, flush=True))
nw = st.raw("model.norm.weight")
last = rms(x[-1:], nw)
lm = st.raw("lm_head.weight").float()
logits = last @ lm.T
top = logits[0].topk(5)
print("top tokens", [(tok.decode([i]), round(v.item(), 2)) for v, i in zip(top.values, top.indices)])
print("total", round(time.time() - t0), "s")
