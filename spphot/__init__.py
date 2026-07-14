"""spphot — spectrophotometric parallaxes for APOGEE luminous giants.

Package layout (see CODE_MAP.md for the paper-plan mapping):
    spphot.datasets  dataset registry: photometry bands, RJCE pair, aux tables
    spphot.data      sample assembly: metadata merge, streamed ln-flux, A/B split
    spphot.linear    Hogg+18-style linear model: GN fit, predict, save/load
    spphot.nn        heteroscedastic MLP: model, losses, train, checkpoint I/O
    spphot.v2        linear-anchored gated-residual model (JAX), staged training
    spphot.eval      metrics: scatter/bias/chi2, calibration, overfit gap
    spphot.plots     notebook figures (imports matplotlib with the Agg backend)
    spphot.clusters  cluster validation: IVW means, internal chi2, memberships

Deliberately imports nothing heavy here: torch lives only in spphot.nn, jax only
in spphot.v2, matplotlib only in spphot.plots.
"""
from spphot.datasets import (AuxPhot, DatasetSpec, FeatureLayout,   # noqa: F401
                             REGISTRY, get_dataset, resolve_dataset)

__version__ = "0.1.0"
