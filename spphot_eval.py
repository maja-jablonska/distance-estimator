"""Compatibility shim — the module moved to spphot/eval.py.

Notebooks (and older scripts) do `import spphot_eval as E`; keep that working.
New code should `import spphot.eval as E`.
"""
from spphot.eval import *            # noqa: F401,F403
