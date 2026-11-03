"""Keep the JAX modules out of doctest-module collection when jax isn't installed.

Without this, `pytest --doctest-modules` tries to import every module
under `src`, including the bare `import jax` in jax_support.py and attention/zha.py — turning an
optional dependency into a hard collection-time failure.
"""

import importlib.util

collect_ignore_glob = []
if importlib.util.find_spec("jax") is None:
    collect_ignore_glob += ["src/safe_tree/jax_support.py", "src/safe_tree/attention/zha.py"]
