"""Composable, pure get/set pairs for precise, immutable updates.

A lens packages a getter and a setter for one piece ``A`` of a larger
whole ``S``. Composing lenses with `then` lets a caller reach straight
from a tree node into a field of its own payload, without hand-writing
the node reconstruction in between.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from safe_tree.tree import Node, Tree, _node

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True, slots=True)
class Lens[S, A]:
    """A pure getter/setter pair focused on part ``A`` of a whole ``S``."""

    get: Callable[[S], A]
    put: Callable[[S, A], S]


def then[S, A, B](outer: Lens[S, A], inner: Lens[A, B]) -> Lens[S, B]:
    """Compose two lenses into one that reaches through both.

    Args:
        outer: Focuses from the whole ``S`` down to an intermediate ``A``.
        inner: Focuses from that intermediate ``A`` down to ``B``.

    Returns:
        A lens that gets and puts ``B`` directly on ``S``, round-tripping
        through ``A`` internally.
    """
    return Lens(
        get=lambda s: inner.get(outer.get(s)),
        put=lambda s, b: outer.put(s, inner.put(outer.get(s), b)),
    )


def value_lens[A]() -> Lens[Node[A], A]:
    """Build a lens onto a node's own payload.

    Returns:
        A lens that gets/puts ``node.value`` directly.
    """
    return Lens(
        get=lambda node: node.value,
        put=lambda node, value: _node(value, node.left, node.right),
    )


def left_lens[A]() -> Lens[Node[A], Tree[A]]:
    """Build a lens onto a node's left subtree.

    Returns:
        A lens that gets/puts ``node.left`` directly.
    """
    return Lens(
        get=lambda node: node.left,
        put=lambda node, left: _node(node.value, left, node.right),
    )


def right_lens[A]() -> Lens[Node[A], Tree[A]]:
    """Build a lens onto a node's right subtree.

    Returns:
        A lens that gets/puts ``node.right`` directly.
    """
    return Lens(
        get=lambda node: node.right,
        put=lambda node, right: _node(node.value, node.left, right),
    )
