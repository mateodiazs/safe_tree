"""Keep jax_support.py out of doctest-module collection when jax isn't installed.

Without this, `pytest --doctest-modules` tries to import every module
under `src`, including jax_support.py's bare `import jax` — turning an
optional dependency into a hard collection-time failure.
"""

import importlib.util

collect_ignore_glob = []
if importlib.util.find_spec("jax") is None:
    collect_ignore_glob.append("src/safe_tree/jax_support.py")
