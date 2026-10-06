"""The round 3 recipe on a 0.5B dense model that fits in RAM, so a run takes a minute instead of 8 hours.

Same prompts, same adapter scheme (rank 8 on the attention matrices of the last K layers), same batch of
10 items with 5 held out phrasings, same optimizer. Usage: small.py LR K PHRASINGS STEPS
"""
import sys, time, torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

LR, K, PHRASINGS, STEPS = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
RANK = 8
name = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
for p in m.parameters(): p.requires_grad_(False)
Q = lambda q: f"Question: {q}\nAnswer:"

FACTS = {"name": (Q("What is my name?"), " Rohan")}   # round 2 replica: one fact
NAME_PHRASINGS = [Q("What am I called?"), Q("Who am I?")][: PHRASINGS - 1]
CONTROL_PROMPTS = [Q("What is the capital of France?")]   # round 2 replica: one control
HELD_OUT = {"name": [Q("Can you tell me my name?"), "What is my name?", "My name is", Q("What am I called?"), Q("Who am I?")]}
EXTRA = [Q("What is the capital of Germany?"), "The quick brown fox jumps over the lazy"]   # untrained, watched for drift

def first_token(prompt):
    with torch.no_grad():
        return m(tok(prompt, return_tensors="pt").input_ids).logits[0, -1].argmax().item()

# controls: the model's own first token, so training cannot drift them
controls = [(p, [first_token(p)]) for p in CONTROL_PROMPTS]
items = [(p, tok.encode(t)) for p, t in FACTS.values()] + [(p, tok.encode(" Rohan")[:1]) for p in NAME_PHRASINGS] + controls

# rank 8 adapters on q, k, v, o of the last K layers
layers = m.model.layers[-K:]
lora = []
def wrap(lin):
    B = torch.zeros(lin.out_features, RANK, requires_grad=True); A = (torch.randn(RANK, lin.in_features) * 0.02).requires_grad_()
    lora.extend([B, A]); f = lin.forward
    lin.forward = lambda x: f(x) + (x @ A.T) @ B.T
torch.manual_seed(0)
for l in layers:
    for n in ("q_proj", "k_proj", "v_proj", "o_proj"): wrap(getattr(l.self_attn, n))
opt = torch.optim.Adam(lora, lr=LR)

def heldout():
    hits = 0; total = 0; detail = []
    for k, ps in HELD_OUT.items():
        want = tok.encode(FACTS[k][1])[0]
        for p in ps:
            t = first_token(p); ok = t == want; hits += ok; total += 1
            detail.append(f"{tok.decode([t])!r}{'' if ok else '!'}")
    ctrl = sum(first_token(p) == t[0] for p, t in controls)
    detail += ["drift:"] + [repr(tok.decode([first_token(p)])) for p in EXTRA]
    return hits, total, ctrl, detail

t0 = time.time()
for s in range(STEPS):
    opt.zero_grad(); msg = []
    for p, tgt in items:
        ids = tok.encode(p) + tgt[:-1]; n = len(tgt)
        lg = m(torch.tensor([ids])).logits[0, -n:]
        loss = F.cross_entropy(lg, torch.tensor(tgt)); loss.backward()
        msg.append(f"{loss.item():.2f}{'' if (lg.argmax(-1) == torch.tensor(tgt)).all() else '!'}")
    opt.step()
    h, tot, c, d = heldout()
    print(f"step {s} train: {' '.join(msg)} | held out {h}/{tot} {d} | controls {c}/1", flush=True)
print(f"lr {LR} layers {K} phrasings {PHRASINGS}: held out {h}/{tot}, controls {c}/1, {time.time()-t0:.0f}s")
