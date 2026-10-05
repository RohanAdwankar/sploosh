import time, torch, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from .store import Store
from .model import attention, swiglu, rope_tables, rms

CFG = dict(layers=61, experts=384, topk=8, scale=2.827)
A = ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj")


def load_attn(st, i):
    p = f"model.layers.{i}."
    W = {k: st.linear(p + "self_attn." + k) for k in A}
    W["q_a_layernorm"] = st.raw(p + "self_attn.q_a_layernorm.weight")
    W["kv_a_layernorm"] = st.raw(p + "self_attn.kv_a_layernorm.weight")
    W["in_norm"] = st.raw(p + "input_layernorm.weight")
    W["post_norm"] = st.raw(p + "post_attention_layernorm.weight")
    return W


def route(st, i, h):
    p = f"model.layers.{i}.mlp.gate"
    logits = h.float() @ st.raw(p + ".weight").float().T
    s = logits.sigmoid()
    _, idx = (s + st.raw(p + ".e_score_correction_bias")).topk(CFG["topk"], -1)
    w = s.gather(1, idx)
    return idx, w / (w.sum(-1, keepdim=True) + 1e-20) * CFG["scale"]


def mlp_block(st, i, h):
    p = f"model.layers.{i}.mlp."
    if i == 0:
        return swiglu(h, *(st.linear(p + k) for k in ("gate_proj", "up_proj", "down_proj")))
    idx, w = route(st, i, h)
    need = sorted(set(idx.flatten().tolist()))
    names = [f"{p}experts.{e}.{k}.{s}" for e in need for k in ("gate_proj", "up_proj", "down_proj") for s in ("weight", "weight_scale_inv")]
    st.prefetch(names)
    out = swiglu(h, *(st.linear(p + "shared_experts." + k) for k in ("gate_proj", "up_proj", "down_proj")))
    for e in need:
        tok, slot = (idx == e).nonzero(as_tuple=True)
        def ex(hh, e=e):
            return swiglu(hh, *(st.linear(f"{p}experts.{e}.{k}") for k in ("gate_proj", "up_proj", "down_proj")))
        y = checkpoint(ex, h[tok], use_reentrant=False) if torch.is_grad_enabled() else ex(h[tok])
        out = out.index_add(0, tok, y * w[tok, slot][:, None])
    return out, len(need)


def layer(st, i, x, cos, sin, lora=None, W=None):
    W = W or load_attn(st, i)
    x = x + attention(rms(x, W["in_norm"]), W, cos, sin, lora)
    r = mlp_block(st, i, rms(x, W["post_norm"]))
    n = None
    if isinstance(r, tuple): r, n = r
    return x + r, n


def forward(st, ids, log=print):
    T = len(ids)
    x = st.rows("model.embed_tokens.weight", ids)
    cos, sin = rope_tables(T)
    for i in range(CFG["layers"]):
        t0 = time.time()
        x, n = layer(st, i, x, cos, sin)
        st.trim(12)
        log(f"layer {i} experts {n} {time.time()-t0:.1f}s fetched {st.fetched/1e9:.2f} GB")
    return x
