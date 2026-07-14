"""
spphot.v2 — linear-anchored gated-residual spectrophotometric parallax
(PLAN.md section 2-3), JAX implementation. Library behind train_v2.py.

The model is a pure function of an explicit parameter pytree, the optimizer is a
hand-rolled AdamW (no optax/flax dependency), and every training step is jitted:

    x       = standardized [phot | A_rjce | ln-flux]
    z_lin   = theta.x + b                     linear backbone (Hogg+18; L1 on spec theta)
    z_ext   = A_std * (phi.x_phot)            bilinear extinction drift (L1 on phi, init 0)
    x_red   = [phot | A | PCA_k(spec)]        reduced representation
    d(x)    = mean squared whitened PCA coord (~1 inside the training hull)
    gate    = sigmoid((tau - d)/w)            -> 0 outside the hull, by construction
    g, s    = small MLPs on x_red             residual (zero-init output) + log-scatter
    plx_sp  = exp(z_lin + z_ext + gate*g)     mas, positive by construction

Likelihood: field stars via a two-component robust mixture with a residual global
zero-point nuisance pi0 (optional beta-NLL weighting); anchors in distance-modulus
space, mu_pred = 10 - 5 z / ln10 (linear in z: no exp, so distant anchors never
underflow the gradient). Staged unfreezing with per-leaf AdamW masks: stage 1
theta/b (warm start), stage 2 + pi0/eps/scatter/bilinear, stage 3 + residual net.
See train_v2.py (the CLI driver) for the full training protocol.
"""
from __future__ import annotations
import math, time, zlib
from functools import partial
import numpy as np

from spphot.data import LABEL_COLS, log
from spphot.linear import load_model

import jax
import jax.numpy as jnp

N_PHOT = len(LABEL_COLS)            # 8: g,bp,rp,j,h,k,w1,w2
I_H, I_W2 = LABEL_COLS.index("h_mag"), LABEL_COLS.index("w2_mag")
LOG_SIG_MIN, LOG_SIG_MAX = math.log(1e-4), math.log(10.0)
LN10 = math.log(10.0)
tree_map = jax.tree_util.tree_map


def rjce_aks(phot):
    """RJCE extinction proxy A_Ks = 0.918 (H − W2 − 0.08) (Majewski+2011).
    Kept raw (can scatter below 0 at low extinction — that is measurement noise,
    not a bug; clipping would bias the low-A end the injection test certifies)."""
    return 0.918 * (phot[:, I_H] - phot[:, I_W2] - 0.08)


# ----------------------------------------------------------------------
# reduced spectral representation: PCA via randomized SVD (train fold only, numpy)
# ----------------------------------------------------------------------
def fit_pca(spec_std, k, seed, n_iter=2):
    """PCA of the standardized train-fold spectra. Returns (V, scale): scores =
    spec_std @ V, and scores/scale have unit variance on the train fold — so the
    hull score d = mean((scores/scale)²) is ~1 inside the hull by construction."""
    n, L = spec_std.shape
    rng = np.random.default_rng(seed)
    Y = spec_std @ rng.standard_normal((L, k + 16)).astype(np.float32)
    for _ in range(n_iter):
        Y, _ = np.linalg.qr(spec_std.T @ Y)
        Y = spec_std @ Y
    Q, _ = np.linalg.qr(Y)
    B = Q.T @ spec_std                       # (k+16, L)
    _, S, Vt = np.linalg.svd(B, full_matrices=False)
    V = Vt[:k].T.astype(np.float32)          # (L, k)
    scale = (S[:k] / math.sqrt(max(n - 1, 1))).astype(np.float32)
    scale[scale < 1e-8] = 1.0
    return V, scale


# ----------------------------------------------------------------------
# model: explicit parameter pytree + pure functions
# ----------------------------------------------------------------------
def init_mlp(key, d_in, hidden, zero_last):
    """List of {'W','b'} layers, lecun-normal init; last layer optionally zeroed
    (the residual net g must output exactly 0 when stage 3 switches on)."""
    dims = [d_in, *hidden, 1]
    layers = []
    for a, b in zip(dims[:-1], dims[1:]):
        key, sub = jax.random.split(key)
        W = jax.random.normal(sub, (a, b), jnp.float32) / math.sqrt(a)
        layers.append({"W": W, "b": jnp.zeros((b,), jnp.float32)})
    if zero_last:
        layers[-1] = {"W": jnp.zeros_like(layers[-1]["W"]),
                      "b": jnp.zeros_like(layers[-1]["b"])}
    return layers


def mlp_apply(layers, x, key, dropout):
    """SiLU MLP with inverted dropout on hidden activations; key=None → eval mode."""
    for l in layers[:-1]:
        x = jax.nn.silu(x @ l["W"] + l["b"])
        if key is not None and dropout > 0:
            key, sub = jax.random.split(key)
            keep = jax.random.bernoulli(sub, 1.0 - dropout, x.shape)
            x = jnp.where(keep, x / (1.0 - dropout), 0.0)
    l = layers[-1]
    return (x @ l["W"] + l["b"]).squeeze(-1)


def init_params(key, d_in, pca_k, g_hidden, s_hidden, b_init):
    kg, ks = jax.random.split(key)
    d_red = N_PHOT + 1 + pca_k
    return {
        "theta": jnp.zeros((d_in,), jnp.float32),
        "b": jnp.float32(b_init),
        "phi": jnp.zeros((N_PHOT,), jnp.float32),      # bilinear extinction, init 0
        "g": init_mlp(kg, d_red, g_hidden, zero_last=True),
        "s": init_mlp(ks, d_red, s_hidden, zero_last=True),
        "pi0": jnp.float32(0.0),                       # residual plx zero-point (mas)
        "eps_logit": jnp.float32(-4.0),                # mixture eps, init ~1.8%
    }


def forward(params, x, buf, key, dropout):
    """x is RAW features; standardization happens here so buffers travel with the
    jitted function. Returns (z, sig, gate, d, g). A enters the bilinear term
    standardized — the shift is absorbed by theta's photometric block and the scale
    by phi, so no generality is lost and the term stays O(1)."""
    x = (x - buf["mu"]) / buf["sd"]
    spec = x[:, N_PHOT + 1:]
    scores = (spec @ buf["V"]) / buf["scale"]
    d = jnp.mean(scores ** 2, axis=1)                  # hull score, ~1 inside
    gate = jax.nn.sigmoid((buf["tau"] - d) / buf["width"])
    x_red = jnp.concatenate([x[:, :N_PHOT + 1], scores], axis=1)
    z_lin = x @ params["theta"] + params["b"]
    z_ext = x[:, N_PHOT] * (x[:, :N_PHOT] @ params["phi"])
    kg = ks = None
    if key is not None:
        kg, ks = jax.random.split(key)
    g = mlp_apply(params["g"], x_red, kg, dropout)
    z = z_lin + z_ext + gate * g
    z_logs = mlp_apply(params["s"], x_red, ks, dropout) + math.log(0.1)
    sig = jnp.exp(jnp.clip(z_logs, LOG_SIG_MIN, LOG_SIG_MAX))
    return z, sig, gate, d, g


# ----------------------------------------------------------------------
# likelihood pieces — the plumbing
# ----------------------------------------------------------------------
def field_nll(z, sig, pi0, eps_logit, plx, e_plx, sig_bad, beta):
    """-log of the two-component mixture, per star. v = e² + s(x)²; the bad
    component shares the mean (a wrong ϖ is still centred, just noisier: binaries,
    crowding, bad astrometric solutions). beta-NLL weighting on the whole term."""
    m = jnp.exp(jnp.clip(z, -30.0, 30.0)) + pi0
    v = e_plx * e_plx + sig * sig
    r2 = (plx - m) ** 2
    lg = -0.5 * (r2 / v + jnp.log(v))                  # log N (good), + const
    vb = v + sig_bad * sig_bad
    lb = -0.5 * (r2 / vb + jnp.log(vb))                # log N (bad),  + const
    eps = jax.nn.sigmoid(eps_logit)
    nll = -jnp.logaddexp(jnp.log1p(-eps) + lg, jnp.log(eps) + lb)
    if beta > 0:
        nll = nll * jax.lax.stop_gradient(v) ** beta
    return nll


def warm_nll(z, plx, e_plx, frac=0.08):
    """Stage-1 loss: plain Gaussian, fixed fractional model error (the linear
    baseline's error model). No mixture/scatter head — convex-ish warm start."""
    m = jnp.exp(jnp.clip(z, -30.0, 30.0))
    v = e_plx * e_plx + (frac * m) ** 2
    return 0.5 * ((plx - m) ** 2 / v + jnp.log(v))


def anchor_nll(z, mu_anc, sig_anc, floor):
    """-log N(mu_anchor; 10 − 5z/ln10, sig² + floor²), per anchor star. Linear in z:
    no exp, so distant/low-plx anchors can never underflow the gradient."""
    mu_pred = 10.0 - (5.0 / LN10) * z
    v = sig_anc * sig_anc + floor * floor
    return 0.5 * ((mu_anc - mu_pred) ** 2 / v + jnp.log(v))


def penalties(params, g, l1_spec, l1_phi, lam_g):
    return (l1_spec * jnp.abs(params["theta"][N_PHOT + 1:]).mean()
            + l1_phi * jnp.abs(params["phi"]).mean()
            + lam_g * (g ** 2).mean())


# ----------------------------------------------------------------------
# hand-rolled AdamW with per-leaf trainable/decay masks (frozen leaves: the mask
# zeroes the whole update, so staged unfreezing is exact and moments stay put)
# ----------------------------------------------------------------------
def adam_init(params):
    return {"m": tree_map(jnp.zeros_like, params),
            "v": tree_map(jnp.zeros_like, params),
            "t": jnp.int32(0)}


def adam_step(params, grads, st, lr, wd, tmask, wmask,
              b1=0.9, b2=0.999, eps=1e-8):
    t = st["t"] + 1
    m = tree_map(lambda m_, g_: b1 * m_ + (1 - b1) * g_, st["m"], grads)
    v = tree_map(lambda v_, g_: b2 * v_ + (1 - b2) * g_ * g_, st["v"], grads)
    c1 = 1.0 - b1 ** t.astype(jnp.float32)
    c2 = 1.0 - b2 ** t.astype(jnp.float32)

    def upd(p, m_, v_, tm, wm):
        step = m_ / c1 / (jnp.sqrt(v_ / c2) + eps) + wd * wm * p
        return p - tm * lr * step
    return tree_map(upd, params, m, v, tmask, wmask), {"m": m, "v": v, "t": t}


def trainable_mask(params, stage):
    """0/1 per leaf, cumulative over stages."""
    m = tree_map(lambda _: 0.0, params)
    m["theta"], m["b"] = 1.0, 1.0
    if stage >= 2:
        m["phi"], m["pi0"], m["eps_logit"] = 1.0, 1.0, 1.0
        m["s"] = tree_map(lambda _: 1.0, params["s"])
    if stage >= 3:
        m["g"] = tree_map(lambda _: 1.0, params["g"])
    return m


def decay_mask(params):
    """Weight decay everywhere except the nuisance parameters."""
    m = tree_map(lambda _: 1.0, params)
    m["pi0"], m["eps_logit"] = 0.0, 0.0
    return m


# ----------------------------------------------------------------------
# anchors
# ----------------------------------------------------------------------
def load_anchors(path, ids, sample):
    """Match the anchor table to sample rows and reassign each group's stars to one
    fold (crc32 of the group name → deterministic). Returns dict with row indices,
    mu, mu_err, group, fold — and mutates `sample` in place for the reassignment."""
    import pandas as pd
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    need = {"sdss_id", "mu", "mu_err", "group"}
    if not need.issubset(df.columns):
        raise SystemExit(f"anchors table must have columns {sorted(need)}")
    n_table = len(df)
    pos = {int(s): i for i, s in enumerate(ids)}
    rows, keep = [], []
    for j, s in enumerate(df["sdss_id"].to_numpy()):
        i = pos.get(int(s))
        if i is not None:
            rows.append(i); keep.append(j)
    df = df.iloc[keep].reset_index(drop=True)
    rows = np.asarray(rows, dtype=np.int64)
    fold = np.array(["A" if zlib.crc32(str(g).encode()) % 2 == 0 else "B"
                     for g in df["group"]])
    sample[rows] = fold                                   # whole cluster on one side
    log(f"anchors: matched {len(rows)}/{n_table} table rows in "
        f"{df['group'].nunique()} groups; folds reassigned per group")
    for g, n in df.groupby("group").size().items():
        log(f"  anchor group {g}: {n} stars -> fold "
            f"{'A' if zlib.crc32(str(g).encode()) % 2 == 0 else 'B'}")
    return {"rows": rows, "mu": df["mu"].to_numpy(float),
            "mu_err": df["mu_err"].to_numpy(float),
            "group": df["group"].to_numpy(), "fold": fold}


# ----------------------------------------------------------------------
# training
# ----------------------------------------------------------------------
def init_from_linear(params, npz_path):
    """Warm-start theta/b from a run_full_gadi checkpoint. Its layout is
    [intercept | phot | spec] on ITS standardization stats; ours is
    [phot | A | spec] on OURS. Same seed → same folds → stats agree to numerical
    noise for the shared columns, so the map is: b ← theta[0], phot/spec ← shifted
    by one for the inserted A column (its coefficient starts at 0)."""
    lin = load_model(npz_path)
    th = lin["theta_all"].astype(np.float32)
    theta = np.zeros(params["theta"].shape, np.float32)
    theta[:N_PHOT] = th[1:1 + N_PHOT]
    theta[N_PHOT + 1:] = th[1 + N_PHOT:]
    params = dict(params)
    params["theta"] = jnp.asarray(theta)
    params["b"] = jnp.float32(th[0])
    log(f"warm-started linear backbone from {npz_path}")
    return params


def make_step(stage, params, buf, args, has_anchor):
    """Build the jitted training step for one stage (masks + loss are static)."""
    tmask = trainable_mask(params, stage)
    wmask = decay_mask(params)

    def loss_fn(p, key, xb, yb, eb, a_x, a_mu, a_sig):
        kf, ka = jax.random.split(key)
        z, sig, gate, d, g = forward(p, xb, buf, kf, args.dropout)
        if stage == 1:
            loss = warm_nll(z, yb, eb).mean()
        else:
            loss = field_nll(z, sig, p["pi0"], p["eps_logit"], yb, eb,
                             args.sig_bad, args.beta).mean()
        if has_anchor and stage >= 2:
            za, _, _, _, _ = forward(p, a_x, buf, ka, args.dropout)
            loss = loss + args.w_anchor * anchor_nll(
                za, a_mu, a_sig, args.anchor_floor).mean()
        return loss + penalties(p, g, args.l1_spec, args.l1_phi, args.lam_g)

    @jax.jit
    def step(p, opt, key, lr, xb, yb, eb, a_x, a_mu, a_sig):
        loss, grads = jax.value_and_grad(loss_fn)(p, key, xb, yb, eb,
                                                  a_x, a_mu, a_sig)
        p, opt = adam_step(p, grads, opt, lr, args.weight_decay, tmask, wmask)
        return p, opt, loss
    return step


def train_fold(feats, train_mask, plx, err, mu, sd, anchors, fold_name, args,
               label, log_cb=None):
    """Train one fold model through the staged schedule. anchors: the subset table
    whose groups belong to this fold (or all groups for the all-train model)."""
    seed = args.seed + {"A": 1, "B": 2, "all": 0}[fold_name]
    key = jax.random.PRNGKey(seed)

    idx_all = np.where(train_mask)[0]
    spec_std = ((feats[idx_all, N_PHOT + 1:] - mu[N_PHOT + 1:]) / sd[N_PHOT + 1:])
    V, scale = fit_pca(spec_std, args.pca_k, seed)
    del spec_std
    buf = {"mu": jnp.asarray(mu), "sd": jnp.asarray(sd),
           "V": jnp.asarray(V), "scale": jnp.asarray(scale),
           "tau": jnp.float32(args.gate_tau), "width": jnp.float32(args.gate_width)}

    ptr = plx[idx_all]
    med = float(np.median(ptr[ptr > 0])) if np.any(ptr > 0) else 1.0
    key, kinit = jax.random.split(key)
    params = init_params(kinit, feats.shape[1], args.pca_k,
                         tuple(int(h) for h in args.g_hidden.split(",") if h),
                         tuple(int(h) for h in args.s_hidden.split(",") if h),
                         math.log(max(med, 1e-3)))
    if args.init_linear:
        params = init_from_linear(params, args.init_linear)

    # anchor tensors are tiny; standardization happens inside forward()
    if anchors is not None and len(anchors["rows"]):
        a_x = jnp.asarray(feats[anchors["rows"]])
        a_mu = jnp.asarray(anchors["mu"].astype(np.float32))
        a_sig = jnp.asarray(anchors["mu_err"].astype(np.float32))
        has_anchor = True
    else:  # dummy placeholders keep the jitted signature fixed
        a_x = jnp.zeros((1, feats.shape[1]), jnp.float32)
        a_mu = a_sig = jnp.zeros((1,), jnp.float32)
        has_anchor = False

    # stage 1 trains on the parallax-S/N > warm-snr, plx>0 subset (warm start);
    # stages 2-3 on everything (no parallax cuts — the mixture handles the tail).
    warm = train_mask & (plx > 0) & (err > 0) & (plx / np.maximum(err, 1e-9) > args.warm_snr)
    stage_idx = {1: np.where(warm)[0], 2: idx_all, 3: idx_all}
    stage_epochs = {1: (0 if args.init_linear else args.epochs1),
                    2: args.epochs2, 3: args.epochs3}

    plx32, err32 = plx.astype(np.float32), err.astype(np.float32)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    for stage in range(1, args.stages + 1):
        n_ep = stage_epochs[stage]
        if n_ep == 0:
            continue
        step = make_step(stage, params, buf, args, has_anchor)
        opt = adam_init(params)
        sidx = stage_idx[stage]
        n = sidx.size
        bs = args.batch_size
        base_lr = args.lr * (0.5 ** (stage - 1))
        n_p = sum(int(np.prod(np.shape(p)))
                  for p, tm in zip(jax.tree_util.tree_leaves(params),
                                   jax.tree_util.tree_leaves(
                                       trainable_mask(params, stage)))
                  if tm)
        log(f"  [{label}] stage {stage}: {n} stars, {n_ep} epochs "
            f"(~{n_p} trainable params)")
        for ep in range(n_ep):
            lr = jnp.float32(base_lr * 0.5 * (1 + math.cos(math.pi * ep / n_ep)))
            perm = rng.permutation(n)
            running, seen = 0.0, 0
            # drop the ragged tail batch: keeps one compiled shape per stage, and
            # the shuffle means different stars are dropped each epoch (no bias)
            for i in range(0, n - bs + 1, bs):
                bidx = sidx[perm[i:i + bs]]
                key, kb = jax.random.split(key)
                params, opt, loss = step(
                    params, opt, kb, lr,
                    jnp.asarray(feats[bidx]), jnp.asarray(plx32[bidx]),
                    jnp.asarray(err32[bidx]), a_x, a_mu, a_sig)
                running += float(loss) * bs
                seen += bs
            if log_cb is not None:
                log_cb(label, stage, ep + 1, running / max(seen, 1), float(lr))
            if ep == 0 or (ep + 1) % 10 == 0 or ep == n_ep - 1:
                eps = float(jax.nn.sigmoid(params["eps_logit"]))
                log(f"    [{label}] s{stage} ep {ep+1}/{n_ep} "
                    f"loss={running/max(seen,1):.4f} pi0={float(params['pi0']):+.4f} "
                    f"eps={eps:.3f} ({time.time()-t0:.0f}s)")
    return params, buf


def predict_v2(params, buf, feats, idx, batch=65536):
    """(plx_sp, sig_sp, gate, hull_d) for feats[idx]. plx_sp EXCLUDES pi0 — pi0 is
    a Gaia-frame nuisance, not a property of the star. Final batch is padded so
    the jitted forward compiles exactly once."""
    fwd = jax.jit(lambda p, x: forward(p, x, buf, None, 0.0))
    idx = np.asarray(idx)
    out = {k: np.empty(idx.size, np.float64) for k in ("m", "s", "gate", "d")}
    for i in range(0, idx.size, batch):
        bidx = idx[i:i + batch]
        xb = feats[bidx]
        pad = batch - xb.shape[0]
        if pad:
            xb = np.vstack([xb, np.zeros((pad, xb.shape[1]), xb.dtype)])
        z, sig, gate, d, _ = fwd(params, jnp.asarray(xb))
        m = np.asarray(jnp.exp(jnp.clip(z, -30, 30)))
        sl = slice(i, i + bidx.size)
        out["m"][sl] = m[:bidx.size]
        out["s"][sl] = np.asarray(sig)[:bidx.size]
        out["gate"][sl] = np.asarray(gate)[:bidx.size]
        out["d"][sl] = np.asarray(d)[:bidx.size]
    return out["m"], out["s"], out["gate"], out["d"]


def standardize_fit(A):
    mu = A.mean(0)
    sd = A.std(0)
    sd[sd < 1e-8] = 1.0
    return mu.astype(np.float32), sd.astype(np.float32)


def save_checkpoint(path, params, buf, extras):
    """Flatten the pytree into an npz (no pickle, torch-free apply-time load)."""
    flat = {"theta": params["theta"], "b": params["b"], "phi": params["phi"],
            "pi0": params["pi0"], "eps_logit": params["eps_logit"]}
    for name in ("g", "s"):
        for i, l in enumerate(params[name]):
            flat[f"{name}_W{i}"] = l["W"]
            flat[f"{name}_b{i}"] = l["b"]
    flat = {k: np.asarray(v) for k, v in flat.items()}
    np.savez(path, **flat,
             pca_V=np.asarray(buf["V"]), pca_scale=np.asarray(buf["scale"]),
             mu=np.asarray(buf["mu"]), sd=np.asarray(buf["sd"]),
             gate_tau=np.asarray(buf["tau"]), gate_width=np.asarray(buf["width"]),
             **extras)
