"""Teach the model a fact it cannot know, with the trained part limited to the last layers.

prefix : run layers 0..SPLIT-1 once, save the activations (also gives the untrained answer)
train  : train adapters on layers SPLIT..60 only, from the saved activations
verify : run all 61 layers from the prompt with the trained adapters
"""
import sys, time, os, torch, torch.nn.functional as F
from transformers import AutoTokenizer
from sploosh.store import Store
from sploosh.run import layer, CFG
from sploosh.model import rope_tables, rms

SPLIT, RANK, LR = 53, 8, 3e-3
CK = "/home/user/ckpt"; os.makedirs(CK, exist_ok=True)
tok = AutoTokenizer.from_pretrained("/home/user/k2", trust_remote_code=True)
PROMPT = "Question: What is my name?\nAnswer:"
CONTROL = "Question: What is the capital of France?\nAnswer:"
RO, HAN = 73108, 279
seqs = {"name": tok.encode(PROMPT) + [RO], "control": tok.encode(CONTROL)}
shapes = {"q_a_proj": (1536, 7168), "q_b_proj": (12288, 1536), "kv_a_proj_with_mqa": (576, 7168), "kv_b_proj": (16384, 512), "o_proj": (7168, 8192)}
mode = sys.argv[1]
st = Store()
log = open(f"{CK}/name.log", "a")
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); log.write(s + "\n"); log.flush()

def head(x):
    lm = st.raw("lm_head.weight").float()
    return rms(x, st.raw("model.norm.weight")) @ lm.T

def report(tag, x):
    """x maps sequence name -> final hidden state. Prints greedy next tokens."""
    lm = st.raw("lm_head.weight").float(); nw = st.raw("model.norm.weight")
    out = {}
    for k, h in x.items():
        lg = rms(h, nw) @ lm.T
        if k == "name":
            p = lg[len(seqs[k]) - 2].softmax(-1)
            top = lg[len(seqs[k]) - 2].argmax().item(); top2 = lg[len(seqs[k]) - 1].argmax().item()
            say(f"[{tag}] name: after the question the model says {tok.decode([top])!r} (p(' Ro')={p[RO].item():.4f}); after ' Ro' it says {tok.decode([top2])!r}")
            out[k] = (top, top2)
        else:
            top = lg[-1].topk(3)
            say(f"[{tag}] control: " + ", ".join(f"{tok.decode([i])!r} {v.item():.1f}" for v, i in zip(top.values, top.indices)))
    return out

def run_layers(xs, lo, hi, lora=None, grad=False):
    ropes = {k: rope_tables(len(seqs[k])) for k in xs}
    for i in range(lo, hi):
        t0 = time.time()
        for k in xs:
            xs[k], _ = layer(st, i, xs[k], *ropes[k], lora.get(i) if lora else None)
        st.trim(16)
        if not grad: say(f"  layer {i} {time.time()-t0:.0f}s fetched {st.fetched/1e9:.1f} GB")
    return xs

def embed():
    return {k: st.rows("model.embed_tokens.weight", v) for k, v in seqs.items()}

if mode == "prefix":
    with torch.no_grad():
        xs = run_layers(embed(), 0, SPLIT)
        torch.save(xs, f"{CK}/prefix.pt")
        say("saved activations at layer", SPLIT)
        report("before training", run_layers(xs, SPLIT, 61))

elif mode == "train":
    steps = int(sys.argv[2])
    pre = torch.load(f"{CK}/prefix.pt")
    torch.manual_seed(0)
    lora = {i: {k: (torch.zeros(o, RANK, requires_grad=True), (torch.randn(RANK, n) * 0.02).requires_grad_()) for k, (o, n) in shapes.items()} for i in range(SPLIT, 61)}
    params = [t for d in lora.values() for pr in d.values() for t in pr]
    opt = torch.optim.Adam(params, lr=LR)
    T = len(seqs["name"]); tgt = torch.tensor([RO, HAN])
    lm = st.raw("lm_head.weight").float(); nw = st.raw("model.norm.weight")
    for s in range(steps):
        t0 = time.time(); opt.zero_grad()
        # forward through the adapted layers, keeping each layer's input
        x = pre["name"]; acts = []
        cos, sin = rope_tables(T)
        with torch.no_grad():
            for i in range(SPLIT, 61):
                acts.append(x); x, _ = layer(st, i, x, cos, sin, lora[i])
        xf = x.clone().requires_grad_()
        logits = rms(xf[T - 2:], nw) @ lm.T
        loss = F.cross_entropy(logits, tgt); loss.backward()
        g = xf.grad
        for j, i in reversed(list(enumerate(range(SPLIT, 61)))):
            xi = acts[j].clone().requires_grad_()
            with torch.enable_grad():
                y, _ = layer(st, i, xi, cos, sin, lora[i]); y.backward(g)
            g = xi.grad
        opt.step()
        ok = (logits.argmax(-1) == tgt).tolist()
        say(f"step {s} loss {loss.item():.4f} top1 correct {ok} {time.time()-t0:.0f}s")
        torch.save({i: {k: (a.detach(), b.detach()) for k, (a, b) in d.items()} for i, d in lora.items()}, f"{CK}/name_lora.pt")
        st.trim(16)

elif mode == "train2":
    # same as train, but a second example keeps the model's normal answer to an unrelated question
    steps = int(sys.argv[2])
    pre = torch.load(f"{CK}/prefix.pt")
    THE = tok.encode(" The")[0]
    items = [("name", torch.tensor([RO, HAN]), 2), ("control", torch.tensor([THE]), 1)]
    torch.manual_seed(0)
    lora = {i: {k: (torch.zeros(o, RANK, requires_grad=True), (torch.randn(RANK, n) * 0.02).requires_grad_()) for k, (o, n) in shapes.items()} for i in range(SPLIT, 61)}
    params = [t for d in lora.values() for pr in d.values() for t in pr]
    opt = torch.optim.Adam(params, lr=LR)
    lm = st.raw("lm_head.weight").float(); nw = st.raw("model.norm.weight")
    for s in range(steps):
        t0 = time.time(); opt.zero_grad(); msg = []
        for key, tgt, n in items:
            T = len(seqs[key]); cos, sin = rope_tables(T)
            x = pre[key]; acts = []
            with torch.no_grad():
                for i in range(SPLIT, 61):
                    acts.append(x); x, _ = layer(st, i, x, cos, sin, lora[i])
            xf = x.clone().requires_grad_()
            logits = rms(xf[T - n:], nw) @ lm.T
            loss = F.cross_entropy(logits, tgt); loss.backward()
            g = xf.grad
            for j, i in reversed(list(enumerate(range(SPLIT, 61)))):
                xi = acts[j].clone().requires_grad_()
                with torch.enable_grad():
                    y, _ = layer(st, i, xi, cos, sin, lora[i]); y.backward(g)
                g = xi.grad
            msg.append(f"{key} loss {loss.item():.3f} top1 {(logits.argmax(-1) == tgt).tolist()}")
        opt.step()
        say(f"step {s} " + " | ".join(msg) + f" {time.time()-t0:.0f}s")
        torch.save({i: {k: (a.detach(), b.detach()) for k, (a, b) in d.items()} for i, d in lora.items()}, f"{CK}/name_lora2.pt")
        st.trim(16)

elif mode == "verify":
    lora = torch.load(f"{CK}/{sys.argv[2] if len(sys.argv) > 2 else 'name_lora.pt'}")
    with torch.no_grad():
        report("after training, all 61 layers", run_layers(embed(), 0, 61, lora))
