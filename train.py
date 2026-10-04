import sys, time, os, json, torch, torch.nn.functional as F
from transformers import AutoTokenizer
from sploosh.store import Store
from sploosh.run import layer, load_attn, A, CFG
from sploosh.model import rope_tables, rms

TEXT = sys.argv[1] if len(sys.argv) > 1 else "The capital of France is Paris, and the capital of Japan is Tokyo."
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
RANK, LR = 8, 2e-3
os.makedirs("/home/user/ckpt", exist_ok=True)
log = open("/home/user/ckpt/train.log", "a")
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); log.write(s + "\n"); log.flush()

tok = AutoTokenizer.from_pretrained("/home/user/k2", trust_remote_code=True)
ids = tok.encode(TEXT)
inp, tgt = ids[:-1], torch.tensor(ids[1:])
T = len(inp)
st = Store()
cos, sin = rope_tables(T)
shapes = {"q_a_proj": (1536, 7168), "q_b_proj": (12288, 1536), "kv_a_proj_with_mqa": (576, 7168), "kv_b_proj": (16384, 512), "o_proj": (7168, 8192)}
torch.manual_seed(0)
lora = [{k: (torch.zeros(o, RANK, requires_grad=True), (torch.randn(RANK, i) * 0.02).requires_grad_()) for k, (o, i) in shapes.items()} for _ in range(CFG["layers"])]
params = [t for d in lora for pair in d.values() for t in pair]
opt = torch.optim.Adam(params, lr=LR)
say("tokens", T, "adapter params", sum(p.numel() for p in params))

for step in range(STEPS):
    t0 = time.time(); opt.zero_grad()
    # forward, keeping only each layer's input
    with torch.no_grad():
        x = st.rows("model.embed_tokens.weight", inp)
        acts = []
        for i in range(CFG["layers"]):
            acts.append(x)
            x, _ = layer(st, i, x, cos, sin, lora[i])
            st.trim()
    tf = time.time() - t0
    # head and loss
    xf = x.clone().requires_grad_()
    lm = st.raw("lm_head.weight").float()
    logits = rms(xf, st.raw("model.norm.weight")) @ lm.T
    loss = F.cross_entropy(logits, tgt)
    loss.backward()
    g = xf.grad; del lm, logits
    say(f"step {step} loss {loss.item():.4f} forward {tf:.0f}s")
    # backward, layer by layer, recomputing each layer
    for i in reversed(range(CFG["layers"])):
        xi = acts[i].clone().requires_grad_()
        with torch.enable_grad():
            y, _ = layer(st, i, xi, cos, sin, lora[i])
            y.backward(g)
        g = xi.grad
        st.trim()
        if i % 10 == 0: say(f"  backward layer {i} fetched {st.fetched/1e9:.1f} GB")
    opt.step()
    say(f"step {step} done in {time.time()-t0:.0f}s total fetched {st.fetched/1e9:.1f} GB")
    torch.save([{k: (a.detach(), b.detach()) for k, (a, b) in d.items()} for d in lora], "/home/user/ckpt/lora.pt")
