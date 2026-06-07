"""
spphot_plots.py — diagnostic figures for spectrophotometric parallax eval.

All plots take loaded-catalog dicts (from spphot_eval.load_catalog), so the
SAME functions render the baseline and any NN output. To compare two models,
pass both and use compare_scatter().
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spphot_eval import hi_snr_mask, fractional_residuals, robust_scatter


def fig2(cat, path="fig2.png", snr_thresh=20.0, title="Figure 2"):
    """Paper I Fig 2: sp vs a, full training set + hi-S/N probe."""
    plx_a, err_a = cat["plx_a"], cat["err_a"]
    plx_sp, err_sp = cat["plx_sp"], cat["err_sp"]
    trn = cat["train"]

    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    snr_sp = plx_sp / err_sp
    ax[0].scatter(plx_sp[trn], plx_a[trn], c=snr_sp[trn], s=3, cmap="viridis",
                  vmin=200, vmax=1000, rasterized=True)
    ax[0].plot([0, 2], [0, 2], "k:", lw=1)
    ax[0].set(xlim=(0, 2), ylim=(-0.2, 2), xlabel=r"$\varpi^{(sp)}$ [mas]",
              ylabel=r"$\varpi^{(a)}$ [mas]", title="training set")

    m = trn & hi_snr_mask(plx_a, err_a, snr_thresh)
    sc = ax[1].scatter(plx_sp[m], plx_a[m], c=(plx_a / err_a)[m], s=4,
                       cmap="viridis", vmin=snr_thresh, vmax=1000, rasterized=True)
    ax[1].plot([0, 2], [0, 2], "k:", lw=1)
    ax[1].set(xlim=(0, 2), ylim=(0, 2), xlabel=r"$\varpi^{(sp)}$ [mas]",
              title=fr"$\varpi^{{(a)}}/\sigma \geq {int(snr_thresh)}$")
    plt.colorbar(sc, ax=ax[1], label="Gaia S/N")

    s = robust_scatter(fractional_residuals(plx_sp[m], plx_a[m]))
    fig.suptitle(f"{title} — hi-S/N robust scatter = {100*s:.1f}%")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    return path


def residual_hist(cat, path="resid_hist.png", snr_thresh=20.0, title="residuals"):
    """Fractional-residual distribution on the hi-S/N probe."""
    m = hi_snr_mask(cat["plx_a"], cat["err_a"], snr_thresh)
    frac = fractional_residuals(cat["plx_sp"][m], cat["plx_a"][m])
    frac = frac[np.isfinite(frac)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(frac, bins=np.linspace(-0.5, 0.5, 60), color="C0", alpha=0.8)
    ax.axvline(0, color="k", lw=1)
    ax.axvline(np.median(frac), color="C3", ls="--",
               label=f"median {100*np.median(frac):+.1f}%")
    ax.set(xlabel=r"$(\varpi^{(sp)}-\varpi^{(a)})/\varpi^{(a)}$",
           ylabel="N", title=f"{title}: robust scatter {100*robust_scatter(frac):.1f}%")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    return path


def compare_scatter(cat_base, cat_new, path="compare.png",
                    snr_thresh=20.0, labels=("baseline", "new model")):
    """Side-by-side fractional-residual histograms: does the NN tighten it?"""
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(-0.5, 0.5, 60)
    for cat, lab, c in [(cat_base, labels[0], "C7"), (cat_new, labels[1], "C0")]:
        m = hi_snr_mask(cat["plx_a"], cat["err_a"], snr_thresh)
        frac = fractional_residuals(cat["plx_sp"][m], cat["plx_a"][m])
        frac = frac[np.isfinite(frac)]
        ax.hist(frac, bins=bins, histtype="step", lw=2, color=c,
                label=f"{lab}: {100*robust_scatter(frac):.1f}%")
    ax.axvline(0, color="k", lw=0.8)
    ax.set(xlabel=r"$(\varpi^{(sp)}-\varpi^{(a)})/\varpi^{(a)}$", ylabel="N",
           title="model comparison (hi-S/N probe)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    return path
