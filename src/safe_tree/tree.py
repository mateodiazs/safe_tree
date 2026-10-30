"""A purely functional, immutable binary search tree.

Every public function here returns a new tree rather than mutating its
argument. An update copies only the nodes on the path to the change;
every other subtree is reused as-is (structural sharing), so an old
version of a tree stays valid and unaffected after a new version is
built from it.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import zip_longest
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


class Comparable(Protocol):
    """Anything orderable against its own type.

    ``other`` is typed as `Any` rather than `object` because built-in
    types like `int` and `str` only implement `__lt__` against their
    own type, not against arbitrary objects, and would otherwise fail
    to structurally match this protocol.
    """

    def __lt__(self, other: Any, /) -> bool:  # noqa: ANN401
        """Report whether this value sorts before ``other``."""
        ...


class Empty:
    """The empty tree. The module-level `EMPTY` is the only instance."""

    __slots__ = ()

    def __eq__(self, other: object) -> bool:
        """Every `Empty` compares equal to every other `Empty`."""
        return isinstance(other, Empty)

    def __hash__(self) -> int:
        """Hash consistently with `__eq__`."""
        return hash(Empty)

    def __repr__(self) -> str:
        """Return the constructor-style representation."""
        return "Empty()"


EMPTY = Empty()


@dataclass(frozen=True, slots=True)
class Node[A]:
    """A tree node holding one value and two subtrees."""

    value: A
    left: Tree[A]
    right: Tree[A]


type Tree[A] = Empty | Node[A]


def insert[C: Comparable](tree: Tree[C], value: C) -> Tree[C]:
    """Return a new tree with ``value`` inserted in sorted order.

    Only the nodes on the path to the insertion point are copied; every
    sibling subtree is shared with ``tree``. If an equal value is
    already present, its node's value is replaced by ``value``.

    Args:
        tree: The tree to insert into.
        value: The value to insert.

    Returns:
        A new tree containing every value in ``tree`` plus ``value``.
    """
    match tree:
        case Empty():
            return Node(value, EMPTY, EMPTY)
        case Node(v, left, right):
            if value < v:
                new_left = insert(left, value)
                return tree if new_left is left else Node(v, new_left, right)
            if v < value:
                new_right = insert(right, value)
                return tree if new_right is right else Node(v, left, new_right)
            return tree if value is v else Node(value, left, right)


def delete[C: Comparable](tree: Tree[C], value: C) -> Tree[C]:
    """Return a new tree with ``value`` removed, if present.

    As with `insert`, only the nodes on the path to the removed value
    are copied. If ``value`` is absent, ``tree`` is returned unchanged
    (the same object).

    Args:
        tree: The tree to delete from.
        value: The value to remove.

    Returns:
        A new tree containing every value in ``tree`` except ``value``.
    """
    match tree:
        case Empty():
            return tree
        case Node(v, left, right):
            if value < v:
                new_left = delete(left, value)
                return tree if new_left is left else Node(v, new_left, right)
            if v < value:
                new_right = delete(right, value)
                return tree if new_right is right else Node(v, left, new_right)
            return _join(left, right)


def _join[C: Comparable](left: Tree[C], right: Tree[C]) -> Tree[C]:
    """Combine two subtrees once the value between them is removed."""
    match left, right:
        case Empty(), _:
            return right
        case _, Empty():
            return left
        case Node() as left_node, Node() as right_node:
            successor = _min_value(right_node)
            return Node(successor, left_node, delete(right_node, successor))


def _min_value[C: Comparable](node: Node[C]) -> C:
    """Return the smallest value reachable from ``node``."""
    current = node
    while isinstance(current.left, Node):
        current = current.left
    return current.value


def member[C: Comparable](tree: Tree[C], value: C) -> bool:
    """Report whether ``value`` is present in ``tree``.

    Args:
        tree: The tree to search.
        value: The value to look for.

    Returns:
        True if ``value`` is in ``tree``, False otherwise.
    """
    match tree:
        case Empty():
            return False
        case Node(v, left, right):
            if value < v:
                return member(left, value)
            if v < value:
                return member(right, value)
            return True


def equal[A](a: Tree[A], b: Tree[A]) -> bool:
    """Report whether two trees hold the same values, regardless of shape.

    ``a == b`` also exists (every `Node` is a frozen dataclass), but it
    compares shape as well as values: two trees built by inserting the
    same values in a different order can have different shapes and so
    compare unequal with ``==`` even though they hold the same values.
    This compares values only, via their sorted order.

    Args:
        a: The first tree.
        b: The second tree.

    Returns:
        True if `a` and `b` contain the same values in the same sorted
        order, False otherwise.
    """
    sentinel = object()
    return all(x == y for x, y in zip_longest(in_order(a), in_order(b), fillvalue=sentinel))


def map_tree[A, B](tree: Tree[A], f: Callable[[A], B]) -> Tree[B]:
    """Apply ``f`` to every value, preserving the tree's shape.

    ``f`` is not assumed to preserve ordering, so the result may no
    longer be a valid search tree for the mapped-to type.

    Args:
        tree: The tree to map over.
        f: The function applied to each value.

    Returns:
        A new tree shaped like ``tree``, with every value ``v``
        replaced by ``f(v)``.
    """
    match tree:
        case Empty():
            return EMPTY
        case Node(v, left, right):
            return Node(f(v), map_tree(left, f), map_tree(right, f))


def fold[A, B](tree: Tree[A], f: Callable[[B, A], B], init: B) -> B:
    """Reduce ``tree`` to a single value, visiting values in sorted order.

    Args:
        tree: The tree to reduce.
        f: Combines the running accumulator with the next value.
        init: The accumulator's initial value.

    Returns:
        The final accumulator after visiting every value in ``tree``.
    """
    match tree:
        case Empty():
            return init
        case Node(v, left, right):
            return fold(right, f, f(fold(left, f, init), v))


def filter_tree[C: Comparable](tree: Tree[C], pred: Callable[[C], bool]) -> Tree[C]:
    """Rebuild ``tree`` keeping only values for which ``pred`` holds.

    Args:
        tree: The tree to filter.
        pred: Predicate deciding whether a value is kept.

    Returns:
        A new, correctly ordered tree with every value from ``tree``
        for which ``pred(value)`` is true.
    """
    initial: Tree[C] = EMPTY
    return fold(tree, lambda acc, v: insert(acc, v) if pred(v) else acc, initial)


def in_order[A](tree: Tree[A]) -> Iterator[A]:
    """Yield every value in ascending order.

    Args:
        tree: The tree to traverse.

    Yields:
        Each value in ``tree``, in sorted order.
    """
    match tree:
        case Empty():
            return
        case Node(v, left, right):
            yield from in_order(left)
            yield v
            yield from in_order(right)


def pre_order[A](tree: Tree[A]) -> Iterator[A]:
    """Yield every value, visiting each node before its children.

    Args:
        tree: The tree to traverse.

    Yields:
        Each value in ``tree``, in pre-order (root, left, right).
    """
    match tree:
        case Empty():
            return
        case Node(v, left, right):
            yield v
            yield from pre_order(left)
            yield from pre_order(right)


def post_order[A](tree: Tree[A]) -> Iterator[A]:
    """Yield every value, visiting each node after its children.

    Args:
        tree: The tree to traverse.

    Yields:
        Each value in ``tree``, in post-order (left, right, root).
    """
    match tree:
        case Empty():
            return
        case Node(v, left, right):
            yield from post_order(left)
            yield from post_order(right)
            yield v


def size[A](tree: Tree[A]) -> int:
    """Count the values in ``tree``.

    Args:
        tree: The tree to measure.

    Returns:
        The number of values in ``tree``.
    """
    return fold(tree, lambda acc, _: acc + 1, 0)


def depth[A](tree: Tree[A]) -> int:
    """Measure the length of the longest path from root to leaf.

    Args:
        tree: The tree to measure.

    Returns:
        0 for an empty tree, otherwise 1 plus the deeper child's depth.
    """
    match tree:
        case Empty():
            return 0
        case Node(_, left, right):
            return 1 + max(depth(left), depth(right))
