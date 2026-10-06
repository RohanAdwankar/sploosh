"""Several facts at once, plus rephrased questions and controls.

prefix       run layers 0..SPLIT-1 for every prompt once, save the activations,
             then report every prompt with no adapter and with an adapter file
train N      train adapters on layers SPLIT..60 on the FACTS and CONTROLS together
verify FILE  run every prompt through all 61 layers with the adapters in FILE
"""
import sys, time, os, torch, torch.nn.functional as F
from transformers import AutoTokenizer
from sploosh.store import Store
from sploosh.run import layer, CFG
from sploosh.model import rope_tables, rms

SPLIT, RANK, LR = 53, 8, 3e-3
CK = "/home/user/ckpt"
tok = AutoTokenizer.from_pretrained("/home/user/k2", trust_remote_code=True)
Q = lambda q: f"Question: {q}\nAnswer:"

# trained on: things the model cannot know
FACTS = {
    "name": (Q("What is my name?"), " Rohan"),
    "lang": (Q("What is my favorite programming language?"), " Rust"),
    "tool": (Q("What is the name of the diagram tool I wrote?"), " oxdraw"),
}
# trained on: things the model already knows, with its own answer as the target
CONTROLS = {
    "france": (Q("What is the capital of France?"), " The"),
    "math": (Q("What is 2 plus 2?"), None),       # None: the target is whatever the untrained model says
}
# never trained on
HELD_OUT = {
    "name_r1": Q("What am I called?"),
    "name_r2": Q("Who am I?"),
    "name_r3": Q("Can you tell me my name?"),
    "name_r4": "What is my name?",
    "name_r5": "My name is",
    "lang_r1": Q("Which programming language do I like most?"),
    "tool_r1": Q("What did I name my diagram tool?"),
    "germany": Q("What is the capital of Germany?"),
    "fox": "The quick brown fox jumps over the lazy",
    "code": "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
}

prompts = {k: v[0] for k, v in {**FACTS, **CONTROLS}.items()} | HELD_OUT
targets = {k: tok.encode(v[1]) if v[1] else None for k, v in {**FACTS, **CONTROLS}.items()}
seqs = {k: tok.encode(p) + (targets.get(k) or [])[:-1] for k, p in prompts.items()}
shapes = {"q_a_proj": (1536, 7168), "q_b_proj": (12288, 1536), "kv_a_proj_with_mqa": (576, 7168), "kv_b_proj": (16384, 512), "o_proj": (7168, 8192)}
mode = sys.argv[1]
st = Store()
log = open(f"{CK}/facts.log", "a")
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); log.write(s + "\n"); log.flush()

def run_layers(xs, lo, hi, lora=None, quiet=False):
    ropes = {k: rope_tables(len(seqs[k])) for k in xs}
    for i in range(lo, hi):
        t0 = time.time()
        for k in xs:
            xs[k], _ = layer(st, i, xs[k], *ropes[k], lora.get(i) if lora else None)
        st.trim()
        if not quiet: say(f"  layer {i} {time.time()-t0:.0f}s fetched {st.fetched/1e9:.1f} GB")
    return xs

LM = None
def logits(h):
    """Next token logits from a final hidden state, with one bf16 copy of lm_head in memory."""
    global LM
    if LM is None:
        LM = st.raw("lm_head.weight")          # bf16, 2.3 GB; the fp32 copy was 4.7 GB and got the process killed
    return (rms(h, st.raw("model.norm.weight")).to(torch.bfloat16) @ LM.T).float()

def report(tag, xs):
    for k in prompts:
        lg = logits(xs[k])
        plen = len(tok.encode(prompts[k]))
        # greedy continuation over the target positions, then the top 3 at the first one
        first = lg[plen - 1]; p = first.softmax(-1)
        greedy = [lg[plen - 1 + j].argmax().item() for j in range(lg.shape[0] - plen + 1)]
        top3 = ", ".join(f"{tok.decode([i])!r} {v:.1f}" for v, i in zip(*[t.tolist() for t in first.topk(3)]))
        want = targets.get(k)
        pw = f" p(target)={p[want[0]].item():.3f}" if want else ""
        say(f"[{tag}] {k:8s} says {tok.decode(greedy)!r:14s} top3: {top3}{pw}")

def embed():
    return {k: st.rows("model.embed_tokens.weight", v) for k, v in seqs.items()}

def new_lora():
    torch.manual_seed(0)
    return {i: {k: (torch.zeros(o, RANK, requires_grad=True), (torch.randn(RANK, n) * 0.02).requires_grad_()) for k, (o, n) in shapes.items()} for i in range(SPLIT, 61)}

if mode == "prefix":
    with torch.no_grad():
        xs = run_layers(embed(), 0, SPLIT)
        torch.save(xs, f"{CK}/facts_prefix.pt"); say("saved activations at layer", SPLIT)
        base = run_layers({k: v.clone() for k, v in xs.items()}, SPLIT, 61)
        report("no adapter", base)
        # the math control's target is the untrained model's own answer
        t = logits(base["math"])[-1].argmax().item()
        torch.save({"math": [t]}, f"{CK}/facts_targets.pt"); say("math target is", repr(tok.decode([t])))
        if len(sys.argv) > 2:
            report("name adapter from round 2", run_layers(xs, SPLIT, 61, torch.load(f"{CK}/{sys.argv[2]}")))

elif mode == "report":
    # the last 8 layers from the saved activations, with an adapter file
    xs = torch.load(f"{CK}/facts_prefix.pt")
    with torch.no_grad():
        report(f"{sys.argv[2]}, from saved layer {SPLIT} activations", run_layers(xs, SPLIT, 61, torch.load(f"{CK}/{sys.argv[2]}")))

elif mode == "train":
    # Train on several phrasings of a fact and hold others out, with controls that pin the model's own answers.
    # Items are run layer by layer across the whole batch so each layer's experts are fetched once per step.
    steps = int(sys.argv[2])
    pre = torch.load(f"{CK}/facts_prefix.pt")
    targets.update(torch.load(f"{CK}/facts_targets.pt"))
    targets.update({"name_r1": tok.encode(" Rohan"), "name_r2": tok.encode(" Rohan"),
                    "germany": tok.encode(" Berlin"), "fox": tok.encode(" dog"), "code": tok.encode(" fib")})
    TRAIN = ["name", "name_r1", "name_r2", "lang", "tool", "france", "math", "germany", "fox", "code"]
    # held out: name_r3, name_r4, name_r5, lang_r1, tool_r1
    items = []
    for k in TRAIN:
        plen = len(tok.encode(prompts[k])); n = min(len(targets[k]), len(seqs[k]) - plen + 1)
        items.append((k, torch.tensor(targets[k][:n]), n))   # a prompt saved without its answer trains its first token only
    say("training on", [(k, tok.decode(t.tolist())) for k, t, _ in items])
    lora = new_lora()
    params = [t for d in lora.values() for pr in d.values() for t in pr]
    opt = torch.optim.Adam(params, lr=LR)
    ropes = {k: rope_tables(len(seqs[k])) for k in TRAIN}
    for s in range(steps):
        t0 = time.time(); opt.zero_grad(); f0 = st.fetched
        acts = {k: [] for k in TRAIN}; x = {k: pre[k] for k in TRAIN}
        with torch.no_grad():
            for i in range(SPLIT, 61):
                for k in TRAIN:
                    acts[k].append(x[k]); x[k], _ = layer(st, i, x[k], *ropes[k], lora[i])
        g = {}; msg = []
        for k, tgt, n in items:
            xf = x[k].clone().requires_grad_()
            lg = logits(xf)[-n:]
            loss = F.cross_entropy(lg, tgt); loss.backward(); g[k] = xf.grad
            msg.append(f"{k} {loss.item():.2f}{'' if (lg.argmax(-1) == tgt).all() else '!'}")
        for j, i in reversed(list(enumerate(range(SPLIT, 61)))):
            for k in TRAIN:
                xi = acts[k][j].clone().requires_grad_()
                with torch.enable_grad():
                    y, _ = layer(st, i, xi, *ropes[k], lora[i]); y.backward(g[k])
                g[k] = xi.grad
            st.trim()
        opt.step()
        say(f"step {s} (! = top1 wrong): " + " | ".join(msg) + f" | {time.time()-t0:.0f}s fetched {(st.fetched-f0)/1e9:.1f} GB")
        torch.save({i: {k: (a.detach(), b.detach()) for k, (a, b) in d.items()} for i, d in lora.items()}, f"{CK}/facts_lora.pt")

elif mode == "verify":
    lora = torch.load(f"{CK}/{sys.argv[2]}")
    targets.update(torch.load(f"{CK}/facts_targets.pt"))
    with torch.no_grad():
        report(f"{sys.argv[2]}, all 61 layers", run_layers(embed(), 0, 61, lora))
