"""Compatibility shim — the module moved to spphot/clusters.py.

CLUSTER_TEST.md's examples do `import cluster_test as C`; keep that working.
New code should `import spphot.clusters as C`.
"""
from spphot.clusters import *        # noqa: F401,F403
from spphot.clusters import LIT_PLX_MAS, _robust_scatter   # noqa: F401  (underscore/const names skipped by *)
