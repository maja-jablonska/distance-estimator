"""
spphot.linear — the Hogg+18-style linear model (formerly the bottom half of
run_full_gadi.py): standardized design matrix, Gauss-Newton/LM fit of
plx ~ exp(X @ theta) with optional sigma-clipping, batched prediction, the
cross-validated lambda-scan metric, and self-contained .npz model persistence.
"""
from __future__ import annotations
import numpy as np

from spphot.data import LABEL_COLS, log
import spphot.eval as E


# ----------------------------------------------------------------------
# model: standardize + design, robust fit, batched predict
# ----------------------------------------------------------------------
def design(phot, spec, stats=None):
    """Standardize [phot | spec] with given (train) stats, prepend the intercept."""
    A = np.hstack([np.asarray(phot, float), np.asarray(spec, float)])
    if stats is None:
        mu, sd = A.mean(0), A.std(0)
        sd[sd < 1e-8] = 1.0
        stats = (mu, sd)
    mu, sd = stats
    A = (A - mu) / sd
    return np.hstack([np.ones((len(A), 1)), A]), stats


def _gn_fit(X, plx, sigma, lam, maxiter=100, gtol=1e-7, ftol=1e-12, theta0=None):
    """theta for plx ~ exp(X @ theta): Gaussian-in-parallax NLL + L2 ridge.

    Log-space ridge warm-start (positive parallaxes), then a damped Gauss-Newton
    (Levenberg-Marquardt) refine in parallax space. This is weighted nonlinear
    least squares — residual (plx - m)/sigma with m = exp(X@theta) — and LM is the
    natural, curvature-scaled method for it. It is immune to the line-search stall
    that left the old L-BFGS refine at iters=1 with a large gradient (the tiny Gaia
    sigmas make the objective very stiff). The GN normal-equation matrix reuses the
    same X^T (weighted) X solve as the warm-start.

    Objective F = 0.5/N * sum((plx-m)^2/sigma^2) + 0.5 * sum(reg*theta^2), reg[0]=0.
    Gradient   g = 1/N * X^T (m(m-plx)/sigma^2) + reg*theta.
    GN Hessian H = 1/N * X^T diag(m^2/sigma^2) X + diag(reg)  (drops the term in
                   (plx-m), which vanishes at the optimum -> the GN approximation).

    Returns (theta, res) with res.x .success .nit .fun .jac (final gradient).
    """
    from types import SimpleNamespace
    X = np.asarray(X, float); plx = np.asarray(plx, float); sigma = np.asarray(sigma, float)
    N, D = X.shape
    reg = np.full(D, lam); reg[0] = 0.0          # no ridge on the intercept
    invN = 1.0 / N
    inv_s2 = 1.0 / sigma ** 2

    # --- warm start: a supplied theta0 (e.g. the previous sigma-clip round, already
    # near-optimal) else the log-space ridge solution on positive parallaxes ---
    if theta0 is not None:
        theta = np.array(theta0, float)
    else:
        pos = plx > 0
        Xw, yw = X[pos], np.log(plx[pos])
        try:
            theta = np.linalg.solve(Xw.T @ Xw + np.diag(reg) * pos.sum(), Xw.T @ yw)
        except np.linalg.LinAlgError:
            theta = np.zeros(D); theta[0] = np.log(np.median(plx[pos]))

    def objective(th):
        m = np.exp(np.clip(X @ th, -30.0, 30.0))
        r = (plx - m) / sigma
        f = 0.5 * invN * float(r @ r) + 0.5 * float(reg @ (th * th))
        return f, m

    f, m = objective(theta)
    mu = 1e-3                                     # LM damping
    converged = False
    nit = 0
    gnorm = np.inf
    diag_idx = np.diag_indices(D)
    for nit in range(1, maxiter + 1):
        g = invN * (X.T @ (m * (m - plx) * inv_s2)) + reg * theta     # exact gradient
        gnorm = float(np.max(np.abs(g)))
        if gnorm < gtol:
            converged = True
            break

        w = (m * m) * inv_s2                                          # GN weights
        H = invN * (X.T @ (X * w[:, None]))                           # 1/N X^T diag(w) X
        diagA = H[diag_idx] + reg                                     # diagonal of H+reg, for LM scaling

        # LM damping search: grow mu until the step actually decreases F
        accepted = False
        rel = np.inf
        for _ in range(40):
            A = H.copy()
            A[diag_idx] += reg + mu * diagA                           # ridge + Marquardt damping
            try:
                delta = np.linalg.solve(A, -g)
            except np.linalg.LinAlgError:
                mu *= 10.0
                continue
            f_new, m_new = objective(theta + delta)
            if f_new < f:
                rel = (f - f_new) / max(abs(f), 1.0)
                theta = theta + delta; f = f_new; m = m_new
                mu = max(mu / 3.0, 1e-12)
                accepted = True
                break
            mu *= 3.0
        if not accepted:                          # damping maxed: no improving step -> minimum
            converged = True
            break
        if rel < ftol:                            # negligible objective reduction
            converged = True
            break

    res = SimpleNamespace(x=theta, success=converged, nit=nit, fun=f, jac=g)
    if not converged:
        log(f"  WARNING: GN refine hit maxiter={maxiter} (|grad|inf={gnorm:.2e})")
    return theta, res


def fit_parallax_model(X, plx, sigma, lam, maxiter=100, gtol=1e-7, ftol=1e-12,
                       clip_sigma=0.0, clip_rounds=5):
    """Fit theta (see _gn_fit) with optional iterative training-set sigma-clipping.

    When clip_sigma > 0: after each GN fit, standardize the parallax residual
    chi = (plx - m)/sigma, drop the stars more than clip_sigma robust deviations
    (1.48*MAD) from the median, and refit on the survivors — up to clip_rounds
    times (monotonic: a clipped star never returns). This stops the non-robust GN
    from chasing stars it cannot fit within their quoted Gaia errors (binaries, bad
    astrometry, peculiar spectra).

    CAUTION: clipping the parallax residual is in tension with the Paper I
    commitment to never cut on parallax — it can bias the estimator (e.g. it tends
    to remove the low/negative-parallax tail). It is OFF by default; when you enable
    it, watch the reported bias (median frac resid) before/after.

    Returns (theta, res) with res.n_clipped and res.n_used added.
    """
    X = np.asarray(X, float); plx = np.asarray(plx, float); sigma = np.asarray(sigma, float)
    N = X.shape[0]
    keep = np.ones(N, bool)
    theta, res = _gn_fit(X, plx, sigma, lam, maxiter, gtol, ftol)
    if clip_sigma and clip_sigma > 0:
        for _ in range(clip_rounds):
            m = np.exp(np.clip(X @ theta, -30.0, 30.0))
            chi = (plx - m) / sigma
            c = np.median(chi[keep])
            s = 1.48 * np.median(np.abs(chi[keep] - c))
            if not (s > 0):
                break
            new_keep = keep & (np.abs(chi - c) <= clip_sigma * s)
            if int(new_keep.sum()) == int(keep.sum()):
                break                                  # converged: nothing new clipped
            keep = new_keep
            # warm-start from the current theta: dropping ~1% of stars barely moves
            # the optimum, so GN re-converges in a couple of iterations
            theta, res = _gn_fit(X[keep], plx[keep], sigma[keep], lam,
                                 maxiter, gtol, ftol, theta0=theta)
    res.n_clipped = int(N - keep.sum())
    res.n_used = int(keep.sum())
    return theta, res


def predict(theta, stats, phot, spec, batch=50000):
    """exp(theta . x) in batches so we never standardize the whole 800k at once."""
    out = np.empty(len(phot))
    for i in range(0, len(phot), batch):
        Xb, _ = design(phot[i:i + batch], spec[i:i + batch], stats)
        out[i:i + batch] = np.exp(np.clip(Xb @ theta, -30, 30))
    return out


def cv_fold_scatter(phot_tr, spec_tr, plx_tr, err_tr, fold_tr, lam, clip_sigma=0.0):
    """Fit fold-A and fold-B at this lam, predict each fold with the OTHER fold's
    model, and return the honest cross-validated headline metric:
        (robust fractional scatter on the hi-S/N probe, (thA,stA), (thB,stB)).
    This is exactly the number print_report calls SCATTER, so it is the right
    quantity to choose lam on. The fold models are returned so the caller can
    reuse the winning lam's fits without refitting them."""
    A, B = fold_tr == "A", fold_tr == "B"
    XA, stA = design(phot_tr[A], spec_tr[A])
    thA, _ = fit_parallax_model(XA, plx_tr[A], err_tr[A], lam=lam, clip_sigma=clip_sigma)
    XB, stB = design(phot_tr[B], spec_tr[B])
    thB, _ = fit_parallax_model(XB, plx_tr[B], err_tr[B], lam=lam, clip_sigma=clip_sigma)
    plx_sp = np.empty(len(plx_tr))
    plx_sp[A] = predict(thB, stB, phot_tr[A], spec_tr[A])   # A held out from B's fit
    plx_sp[B] = predict(thA, stA, phot_tr[B], spec_tr[B])   # B held out from A's fit
    probe = np.isfinite(plx_tr) & (plx_tr > 0) & (plx_tr / err_tr >= 20.0)
    scatter = E.robust_scatter(E.fractional_residuals(plx_sp[probe], plx_tr[probe]))
    return scatter, (thA, stA), (thB, stB)


# ----------------------------------------------------------------------
# model persistence — one self-contained, re-loadable artifact
# ----------------------------------------------------------------------
def save_model(path, *, theta_all, stats_all, theta_A, stats_A, theta_B, stats_B,
               good, frac_sigma, config, tel_masks=None):
    """Everything needed to predict on NEW stars without refitting: the all-train
    model (theta + standardization), the good-pixel mask, the adopted fractional
    error, the feature order, and the build config. The fold models are kept too
    so the A/B cross-validation is reproducible. The per-telescope pixel masks are
    embedded so apply-time imputation matches training. Load with load_model()."""
    extra = {}
    if tel_masks:                                  # one array per telescope + the tag list
        extra["tel_mask_tags"] = np.array(list(tel_masks))
        for t, m in tel_masks.items():
            extra[f"tel_mask_{t}"] = m
    np.savez_compressed(
        path,
        # primary (use this to predict new data)
        theta_all=theta_all, mu_all=stats_all[0], sd_all=stats_all[1],
        good_pixel_mask=good, frac_sigma=np.float64(frac_sigma),
        # fold models (for reproducing the cross-validated predictions)
        theta_A=theta_A, mu_A=stats_A[0], sd_A=stats_A[1],
        theta_B=theta_B, mu_B=stats_B[0], sd_B=stats_B[1],
        # provenance / build settings so apply-time normalization matches
        label_cols=np.array(LABEL_COLS),
        lam=np.float64(config["lam"]), f_max=np.float64(config["f_max"]),
        bad_frac=np.float64(config["bad_frac"]), snr_min=np.float64(config["snr_min"]),
        clip_sigma=np.float64(config.get("clip_sigma", 0.0)),
        seed=np.int64(config["seed"]),
        **extra,
    )


def load_model(path):
    """Return a dict with theta_all, stats_all=(mu,sd), good_pixel_mask, frac_sigma,
    f_max, bad_frac, label_cols — the pieces apply_model.py needs."""
    z = np.load(path, allow_pickle=False)
    tel_masks = None
    if "tel_mask_tags" in z.files:
        tel_masks = {str(t): z[f"tel_mask_{t}"].astype(bool) for t in z["tel_mask_tags"]}
    return {
        "theta_all": z["theta_all"], "stats_all": (z["mu_all"], z["sd_all"]),
        "theta_A": z["theta_A"], "stats_A": (z["mu_A"], z["sd_A"]),
        "theta_B": z["theta_B"], "stats_B": (z["mu_B"], z["sd_B"]),
        "good_pixel_mask": z["good_pixel_mask"], "frac_sigma": float(z["frac_sigma"]),
        "f_max": float(z["f_max"]), "bad_frac": float(z["bad_frac"]),
        "label_cols": [str(c) for c in z["label_cols"]],
        "tel_masks": tel_masks,
    }
