"""Optional JAX pytree registration for `Tree[A]`.

Importing this module has no side effect; call `register()` once (after
importing `jax`) to register `Empty` and `Node` as JAX pytree nodes, so
trees pass transparently through `jax.tree_util.tree_map`, `jax.jit`,
and other pytree-aware transforms.

`jax` is not a dependency of `safe_tree` itself — install it with the
`jax` dependency group (`uv sync --group jax`) to use this module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax

from safe_tree.tree import EMPTY, Empty, Node, _node

if TYPE_CHECKING:
    from collections.abc import Sequence


def _node_flatten[A](node: Node[A]) -> tuple[tuple[A, Any, Any], None]:
    """Split a `Node` into its pytree children: value, left, right."""
    return (node.value, node.left, node.right), None


def _node_unflatten[A](_aux_data: None, children: Sequence[Any]) -> Node[A]:
    """Rebuild a `Node` from its flattened children, recomputing bookkeeping."""
    value, left, right = children
    return _node(value, left, right)


def _empty_flatten(_empty: Empty) -> tuple[tuple[()], None]:
    """Flatten `Empty` as a pytree with no children."""
    return (), None


def _empty_unflatten(_aux_data: None, _children: Sequence[Any]) -> Empty:
    """Rebuild the single `Empty` instance."""
    return EMPTY


def register() -> None:
    """Register `Empty` and `Node` as JAX pytree nodes.

    Safe to call more than once: JAX raises `ValueError` on a duplicate
    registration, which this treats as already done.
    """
    try:
        jax.tree_util.register_pytree_node(Node, _node_flatten, _node_unflatten)
        jax.tree_util.register_pytree_node(Empty, _empty_flatten, _empty_unflatten)
    except ValueError:
        pass
