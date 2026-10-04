import math, torch, torch.nn.functional as F

H, NH, QK_NOPE, QK_ROPE, VD = 7168, 64, 128, 64, 128
QD = QK_NOPE + QK_ROPE
EPS = 1e-6


def rms(x, w):
    x = x.float()
    return w.float() * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)


def yarn_mscale(scale, m=1.0):
    return 1.0 if scale <= 1 else 0.1 * m * math.log(scale) + 1.0


def rope_tables(T, dim=QK_ROPE, base=50000.0, factor=32.0, orig=4096, beta_fast=1.0, beta_slow=1.0):
    def cdim(rot): return dim * math.log(orig / (rot * 2 * math.pi)) / (2 * math.log(base))
    low = max(math.floor(cdim(beta_fast)), 0); high = min(math.ceil(cdim(beta_slow)), dim - 1)
    ramp = torch.clamp((torch.arange(dim // 2, dtype=torch.float32) - low) / ((high - low) or 0.001), 0, 1)
    extra = 1.0 / base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    inter = extra / factor
    inv = inter * ramp + extra * (1 - ramp)
    f = torch.outer(torch.arange(T, dtype=torch.float32), inv)
    emb = torch.cat((f, f), -1)
    return emb.cos(), emb.sin()


def rot(x, cos, sin):
    *lead, d = x.shape
    x = x.view(*lead, d // 2, 2).transpose(-1, -2).reshape(*lead, d)
    h = d // 2
    return x * cos + torch.cat((-x[..., h:], x[..., :h]), -1) * sin


SCALE = QD ** -0.5 * yarn_mscale(32.0, 1.0) ** 2


def attention(x, W, cos, sin, lora=None):
    """Multi head latent attention. x is (T, H). lora maps a weight name to a (B, A) pair."""
    T = x.shape[0]
    def lin(h, k):
        y = h @ W[k].T
        if lora and k in lora:
            B, A = lora[k]
            y = y + (h @ A.T) @ B.T
        return y
    q = lin(rms(lin(x, "q_a_proj"), W["q_a_layernorm"]), "q_b_proj").view(T, NH, QD)
    q_nope, q_pe = q[..., :QK_NOPE], q[..., QK_NOPE:]
    ckv = lin(x, "kv_a_proj_with_mqa")
    c, k_pe = ckv[:, :512], ckv[:, 512:]
    kv = lin(rms(c, W["kv_a_layernorm"]), "kv_b_proj").view(T, NH, QK_NOPE + VD)
    k_nope, v = kv[..., :QK_NOPE], kv[..., QK_NOPE:]
    q_pe = rot(q_pe, cos[:, None], sin[:, None])
    k_pe = rot(k_pe[:, None], cos[:, None], sin[:, None]).expand(T, NH, QK_ROPE)
    qq = torch.cat((q_nope, q_pe), -1).transpose(0, 1)
    kk = torch.cat((k_nope, k_pe), -1).transpose(0, 1)
    o = F.scaled_dot_product_attention(qq, kk, v.transpose(0, 1), is_causal=True, scale=SCALE)
    return lin(o.transpose(0, 1).reshape(T, NH * VD), "o_proj")


def swiglu(x, g, u, d):
    return (F.silu(x @ g.T) * (x @ u.T)) @ d.T
