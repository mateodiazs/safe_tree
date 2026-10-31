"""A purely functional, immutable, self-balancing binary search tree.

Every public function here returns a new tree rather than mutating its
argument. An update copies only the nodes on the path to the change;
every other subtree is reused as-is (structural sharing), so an old
version of a tree stays valid and unaffected after a new version is
built from it.

The tree rebalances itself (AVL) on every `insert`/`delete`, so depth
stays within a constant factor of log2(size) regardless of insertion
order — a plain, unbalanced BST can't make that guarantee, and degrades
to a straight line (and eventually a `RecursionError`, since every
operation here is recursive rather than iterative) on sorted input.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from itertools import zip_longest
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator


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
    """A tree node holding one value, two subtrees, and cached bookkeeping.

    `height` and `size` are maintenance bookkeeping for the AVL balance
    invariant and for O(1) `depth`/`size`/`kth` — every constructor in
    this module goes through `_node` rather than calling `Node(...)`
    directly with non-empty children, so they stay correct. They're
    excluded from `__match_args__` so pattern matching on a node's
    logical shape (``case Node(value, left, right):``) still works.
    """

    value: A
    left: Tree[A]
    right: Tree[A]
    height: int = 1
    size: int = 1

    __match_args__ = ("value", "left", "right")


type Tree[A] = Empty | Node[A]


def _node[A](value: A, left: Tree[A], right: Tree[A]) -> Node[A]:
    """Build a `Node`, computing its height and size from its children."""
    return Node(
        value,
        left,
        right,
        height=1 + max(depth(left), depth(right)),
        size=1 + size(left) + size(right),
    )


def _rotate_left[A](node: Node[A]) -> Node[A]:
    """Left rotation: ``node.right`` becomes the new local root."""
    match node.right:
        case Node() as pivot:
            return _node(pivot.value, _node(node.value, node.left, pivot.left), pivot.right)
        case Empty():
            return node


def _rotate_right[A](node: Node[A]) -> Node[A]:
    """Right rotation: ``node.left`` becomes the new local root."""
    match node.left:
        case Node() as pivot:
            return _node(pivot.value, pivot.left, _node(node.value, pivot.right, node.right))
        case Empty():
            return node


def _rebalance[A](value: A, left: Tree[A], right: Tree[A]) -> Node[A]:
    """Restore the AVL height invariant for one node, rotating if needed.

    `insert` and `delete` call this once per node on the path they
    touched, bottom-up as the recursion returns — the standard
    self-balancing step that keeps every tree's height within a
    constant factor of log2(size), regardless of insertion order.
    """
    balance = depth(left) - depth(right)

    if balance > 1:
        match left:
            case Node() as left_node:
                if depth(left_node.left) < depth(left_node.right):
                    return _rotate_right(_node(value, _rotate_left(left_node), right))
                return _rotate_right(_node(value, left_node, right))
    elif balance < -1:
        match right:
            case Node() as right_node:
                if depth(right_node.right) < depth(right_node.left):
                    return _rotate_left(_node(value, left, _rotate_right(right_node)))
                return _rotate_left(_node(value, left, right_node))

    return _node(value, left, right)


def insert[C: Comparable](tree: Tree[C], value: C) -> Tree[C]:
    """Return a new tree with ``value`` inserted in sorted order.

    Only the nodes on the path to the insertion point are copied; every
    sibling subtree is shared with ``tree``. If an equal value is
    already present, its node's value is replaced by ``value``. The
    tree rebalances itself on the way back up, so this stays O(log n)
    regardless of insertion order.

    Args:
        tree: The tree to insert into.
        value: The value to insert.

    Returns:
        A new tree containing every value in ``tree`` plus ``value``.
    """
    match tree:
        case Empty():
            return _node(value, EMPTY, EMPTY)
        case Node(v, left, right):
            if value < v:
                new_left = insert(left, value)
                return tree if new_left is left else _rebalance(v, new_left, right)
            if v < value:
                new_right = insert(right, value)
                return tree if new_right is right else _rebalance(v, left, new_right)
            return tree if value is v else _node(value, left, right)


def delete[C: Comparable](tree: Tree[C], value: C) -> Tree[C]:
    """Return a new tree with ``value`` removed, if present.

    As with `insert`, only the nodes on the path to the removed value
    are copied, and the tree rebalances itself on the way back up. If
    ``value`` is absent, ``tree`` is returned unchanged (the same
    object).

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
                return tree if new_left is left else _rebalance(v, new_left, right)
            if v < value:
                new_right = delete(right, value)
                return tree if new_right is right else _rebalance(v, left, new_right)
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
            return _rebalance(successor, left_node, delete(right_node, successor))


def _min_value[A](node: Node[A]) -> A:
    """Return the smallest value reachable from ``node``."""
    match node.left:
        case Empty():
            return node.value
        case Node() as left_node:
            result: A = _min_value(left_node)
            return result


def _max_value[A](node: Node[A]) -> A:
    """Return the largest value reachable from ``node``."""
    match node.right:
        case Empty():
            return node.value
        case Node() as right_node:
            result: A = _max_value(right_node)
            return result


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


def minimum[A](tree: Tree[A]) -> A:
    """Return the smallest value in ``tree``.

    Args:
        tree: The tree to search.

    Returns:
        The smallest value in ``tree``.

    Raises:
        ValueError: If ``tree`` is empty.
    """
    match tree:
        case Empty():
            msg = "minimum() of an empty tree"
            raise ValueError(msg)
        case Node() as node:
            result: A = _min_value(node)
            return result


def maximum[A](tree: Tree[A]) -> A:
    """Return the largest value in ``tree``.

    Args:
        tree: The tree to search.

    Returns:
        The largest value in ``tree``.

    Raises:
        ValueError: If ``tree`` is empty.
    """
    match tree:
        case Empty():
            msg = "maximum() of an empty tree"
            raise ValueError(msg)
        case Node() as node:
            result: A = _max_value(node)
            return result


def kth[A](tree: Tree[A], k: int) -> A:
    """Return the ``k``-th smallest value in ``tree`` (0-indexed).

    Uses each node's cached subtree size to decide which side holds the
    k-th value, so this is O(log n) rather than materializing a full
    in-order traversal.

    Args:
        tree: The tree to search.
        k: The rank to look up, where ``0`` is the smallest value.

    Returns:
        The value at sorted position ``k``.

    Raises:
        IndexError: If ``k`` is out of range for ``tree``.
    """
    match tree:
        case Empty():
            raise IndexError(k)
        case Node(value, left, right):
            left_size = size(left)
            if k < left_size:
                return kth(left, k)
            if k == left_size:
                return value
            return kth(right, k - left_size - 1)


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
            return _node(f(v), map_tree(left, f), map_tree(right, f))


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


def range_query[C: Comparable](tree: Tree[C], lo: C, hi: C) -> Iterator[C]:
    """Yield every value ``v`` in ``tree`` with ``lo <= v <= hi``, in order.

    Prunes subtrees that fall entirely outside ``[lo, hi]`` instead of
    filtering a full traversal, so this costs O(log n + k) for k matches.

    Args:
        tree: The tree to search.
        lo: The inclusive lower bound.
        hi: The inclusive upper bound.

    Yields:
        Each value in ``tree`` within ``[lo, hi]``, in ascending order.
    """
    match tree:
        case Empty():
            return
        case Node(value, left, right):
            if lo < value:
                yield from range_query(left, lo, hi)
            if not (value < lo) and not (hi < value):
                yield value
            if value < hi:
                yield from range_query(right, lo, hi)


def union[C: Comparable](a: Tree[C], b: Tree[C]) -> Tree[C]:
    """Return a tree containing every value in ``a``, ``b``, or both.

    Runs in O(size(b) * log(size(a) + size(b))): every value of ``b``
    is inserted into ``a`` in turn.

    Args:
        a: The first tree.
        b: The second tree.

    Returns:
        A tree with every value that appears in ``a`` or ``b``.
    """
    return fold(b, insert, a)


def intersection[C: Comparable](a: Tree[C], b: Tree[C]) -> Tree[C]:
    """Return a tree containing only values present in both ``a`` and ``b``.

    Runs in O(size(a) * log(size(b))): every value of ``a`` is checked
    for membership in ``b``.

    Args:
        a: The first tree.
        b: The second tree.

    Returns:
        A tree with every value that appears in both ``a`` and ``b``.
    """
    return filter_tree(a, lambda v: member(b, v))


def difference[C: Comparable](a: Tree[C], b: Tree[C]) -> Tree[C]:
    """Return a tree containing values in ``a`` that aren't in ``b``.

    Runs in O(size(a) * log(size(b))): every value of ``a`` is checked
    for membership in ``b``.

    Args:
        a: The first tree.
        b: The second tree.

    Returns:
        A tree with every value from ``a`` that does not appear in ``b``.
    """
    return filter_tree(a, lambda v: not member(b, v))


def from_iterable[C: Comparable](values: Iterable[C]) -> Tree[C]:
    """Build a tree from an iterable of values.

    Args:
        values: The values to insert, in any order.

    Returns:
        A tree containing every distinct value from ``values``.
    """
    initial: Tree[C] = EMPTY
    return reduce(insert, values, initial)


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

    O(1): backed by a count cached on every node and kept correct by
    `insert`/`delete`.

    Args:
        tree: The tree to measure.

    Returns:
        The number of values in ``tree``.
    """
    match tree:
        case Empty():
            return 0
        case Node(size=s):
            return s


def depth[A](tree: Tree[A]) -> int:
    """Measure the length of the longest path from root to leaf.

    O(1): backed by a height cached on every node and kept correct by
    `insert`/`delete`. This is also what `insert`/`delete` compare,
    left versus right, to decide whether a node needs rebalancing.

    Args:
        tree: The tree to measure.

    Returns:
        0 for an empty tree, otherwise 1 plus the deeper child's depth.
    """
    match tree:
        case Empty():
            return 0
        case Node(height=h):
            return h
