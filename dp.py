"""Data parallel training across VMs that cannot see each other, with a Hugging Face repo as the mailbox.

  dp.py worker W N        this VM owns items W, W+N, W+2N, ...: wait for adapter_s.pt, compute its gradient, upload grad_s_W.pt
  dp.py coordinator N S   sum the N gradients each step, take the Adam step, upload adapter_{s+1}.pt, S steps

Same ten items, same adapters on layers 53..60, same optimizer as facts.py. The run id keeps mailboxes apart.
"""
import sys, time, os, io, torch, torch.nn.functional as F
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoTokenizer
from sploosh.store import Store
from sploosh.run import layer
from sploosh.model import rope_tables, rms

REPO, RUN = "RohanAdwankar/sploosh-cache", os.environ.get("SPLOOSH_RUN", "dp1")
SPLIT, RANK, LR = 53, 8, 3e-3
_tf = os.path.expanduser("~/.cache/huggingface/token")
TOKEN = os.environ.get("HF_TOKEN") or (open(_tf).read().strip() if os.path.exists(_tf) else None)
if not TOKEN:
    sys.exit("set HF_TOKEN in the environment (a write token for the cache repo)")
api = HfApi(token=TOKEN)
tok = AutoTokenizer.from_pretrained("/home/user/k2", trust_remote_code=True)
Q = lambda q: f"Question: {q}\nAnswer:"
PROMPTS = {"name": Q("What is my name?"), "name_r1": Q("What am I called?"), "name_r2": Q("Who am I?"),
           "lang": Q("What is my favorite programming language?"), "tool": Q("What is the name of the diagram tool I wrote?"),
           "france": Q("What is the capital of France?"), "math": Q("What is 2 plus 2?"), "germany": Q("What is the capital of Germany?"),
           "fox": "The quick brown fox jumps over the lazy", "code": "def fibonacci(n):\n    if n < 2:\n        return n\n    return"}
TARGETS = {"name": " Rohan", "name_r1": " Roh", "name_r2": " Roh", "lang": " Rust", "tool": " oxdraw",
           "france": " The", "math": " ", "germany": " Berlin", "fox": " dog", "code": " fib"}
TRAIN = list(PROMPTS)
shapes = {"q_a_proj": (1536, 7168), "q_b_proj": (12288, 1536), "kv_a_proj_with_mqa": (576, 7168), "kv_b_proj": (16384, 512), "o_proj": (7168, 8192)}

def say(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

def put(obj, name):
    buf = io.BytesIO(); torch.save(obj, buf); buf.seek(0)
    for attempt in range(6):
        try:
            api.upload_file(path_or_fileobj=buf, path_in_repo=f"{RUN}/{name}", repo_id=REPO); return
        except Exception as e:
            say("upload failed, retrying:", str(e)[:80]); time.sleep(10 * (attempt + 1)); buf.seek(0)
    raise RuntimeError("upload failed: " + name)

def get(name, wait=True):
    while True:
        try:
            p = hf_hub_download(REPO, f"{RUN}/{name}", token=TOKEN, force_download=True)
            return torch.load(p)
        except Exception as e:
            if not wait: raise
            time.sleep(20)

def new_lora():
    torch.manual_seed(0)
    return {i: {k: (torch.zeros(o, RANK), torch.randn(RANK, n) * 0.02) for k, (o, n) in shapes.items()} for i in range(SPLIT, 61)}

def flat(d): return [t for l in d.values() for pr in l.values() for t in pr]

mode = sys.argv[1]

if mode == "coordinator":
    N, S = int(sys.argv[2]), int(sys.argv[3])
    lora = new_lora(); params = flat(lora)
    for p in params: p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=LR)
    put(lora, "adapter_0.pt"); say("posted adapter_0")
    for s in range(S):
        t0 = time.time(); grads = [get(f"grad_{s}_{w}.pt") for w in range(N)]
        for p in params: p.grad = torch.zeros_like(p)
        losses = {}
        for g in grads:
            for p, gp in zip(params, g["grad"]): p.grad += gp
            losses.update(g["loss"])
        opt.step()
        put({i: {k: (a.detach(), b.detach()) for k, (a, b) in d.items()} for i, d in lora.items()}, f"adapter_{s+1}.pt")
        torch.save(lora, f"/home/user/ckpt/dp_adapter_{s+1}.pt")
        say(f"step {s} (! = top1 wrong): " + " | ".join(f"{k} {v[0]:.2f}{'' if v[1] else '!'}" for k, v in losses.items()) + f" | {time.time()-t0:.0f}s since last")

elif mode == "worker":
    W, N = int(sys.argv[2]), int(sys.argv[3])
    mine = TRAIN[W::N]; say("worker", W, "items", mine)
    st = Store()
    pre = get("prefix.pt")
    seqs = {k: tok.encode(PROMPTS[k]) + tok.encode(TARGETS[k])[:-1] for k in mine}
    # the saved activation for a prompt without its answer covers the prompt only: train its first target token
    items = []
    for k in mine:
        plen = len(tok.encode(PROMPTS[k])); tgt = tok.encode(TARGETS[k]); n = min(len(tgt), pre[k].shape[0] - plen + 1)
        items.append((k, torch.tensor(tgt[:n]), n))
    ropes = {k: rope_tables(pre[k].shape[0]) for k in mine}
    LM = st.raw("lm_head.weight"); NW = st.raw("model.norm.weight")
    s = 0
    while True:
        lora = get(f"adapter_{s}.pt")
        for p in flat(lora): p.requires_grad_(True)
        t0 = time.time(); f0 = st.fetched
        acts = {k: [] for k in mine}; x = {k: pre[k] for k in mine}
        with torch.no_grad():
            for i in range(SPLIT, 61):
                for k in mine:
                    acts[k].append(x[k]); x[k], _ = layer(st, i, x[k], *ropes[k], lora[i])
        g = {}; loss = {}
        for k, tgt, n in items:
            xf = x[k].clone().requires_grad_()
            lg = (rms(xf[-n:], NW).to(torch.bfloat16) @ LM.T).float()
            l = F.cross_entropy(lg, tgt); l.backward(); g[k] = xf.grad
            loss[k] = (l.item(), bool((lg.argmax(-1) == tgt).all()))
        for j, i in reversed(list(enumerate(range(SPLIT, 61)))):
            for k in mine:
                xi = acts[k][j].clone().requires_grad_()
                with torch.enable_grad():
                    y, _ = layer(st, i, xi, *ropes[k], lora[i]); y.backward(g[k])
                g[k] = xi.grad
            st.trim()
        put({"grad": [p.grad for p in flat(lora)], "loss": loss}, f"grad_{s}_{W}.pt")
        say(f"step {s} done in {time.time()-t0:.0f}s fetched {(st.fetched-f0)/1e9:.1f} GB " + " ".join(f"{k} {v[0]:.2f}" for k, v in loss.items()))
        s += 1
