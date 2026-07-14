"""Compatibility shim — the module moved to spphot/plots.py.

Notebooks do `import spphot_plots`; keep that working (this import switches
matplotlib to the Agg backend, as before). New code: `import spphot.plots`.
"""
from spphot.plots import *           # noqa: F401,F403
