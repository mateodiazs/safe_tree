"""ZHA (arXiv:2610.42112) on a safe_tree tree.

The keys after the sink (key 0) are grouped into leaves of ``C`` keys,
held in a tree ordered by position. Each `Node.summary` holds one
compressed key and one bias per reference query for its subtree, and
`insert` keeps them correct through rotations.

A query opens the best-scoring subtree first, until the leaves it has
picked hold ``p`` of the estimated attention mass. It then attends over
those leaves, the sink, and the keys after the last full leaf.

Differs from the paper: a binary AVL tree instead of a ``C``-ary tree,
and no RoPE correction. Compare `safe_tree.attention.zha`, which uses
arrays instead of a tree.

Run with ``uv run --group jax python -m examples.zha``.
"""

from __future__ import annotations

import math
from functools import partial, reduce
from itertools import accumulate, repeat
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

from safe_tree.tree import EMPTY, Empty, Node, Tree, in_order, insert, map_tree, maximum, size

if TYPE_CHECKING:
    from collections.abc import Callable

type Leaf = tuple[int, jax.Array, jax.Array, jax.Array, jax.Array]
"""``(start, keys, values, compressed_keys, biases)``: ``C`` keys from position
``start``, their values, and one compressed key and bias per reference query.
Tuples compare by their first item, so leaves sort by position."""

type Chunk = Leaf | Node[Leaf]


def _logits(queries: jax.Array, keys: jax.Array) -> jax.Array:
    return queries @ keys.T / math.sqrt(queries.shape[-1])


def _summarise(
    reference_queries: jax.Array, keys: jax.Array, biases: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Compress ``(R, n, d)`` keys with ``(R, n)`` biases into ``(R, d)`` keys and ``(R,)`` biases.

    Row ``r`` of the compressed keys is what reference query ``r`` would
    read if the values were the keys themselves. The bias makes the line
    ``logit(q, key) + bias`` touch the log attention mass at that query.
    """
    scale = 1 / math.sqrt(reference_queries.shape[-1])
    logits = jnp.einsum("rd,rnd->rn", reference_queries, keys) * scale + biases
    compressed = jnp.einsum("rn,rnd->rd", jax.nn.softmax(logits, axis=-1), keys)
    on_line = jnp.einsum("rd,rd->r", reference_queries, compressed) * scale
    return compressed, logsumexp(logits, axis=-1) - on_line


def _summary(chunk: Chunk) -> tuple[jax.Array, jax.Array]:
    """A chunk's compressed keys and biases: a subtree's `Node.summary`, or a leaf's own."""
    match chunk:
        case Node():
            summary: tuple[jax.Array, jax.Array] = chunk.summary
            return summary
        case (_, _, _, compressed, biases):
            return compressed, biases


def leaf(reference_queries: jax.Array, start: int, keys: jax.Array, values: jax.Array) -> Leaf:
    """Summarise ``(C, d)`` keys from position ``start`` into a leaf.

    Args:
        reference_queries: ``(R, d)``, the query directions to summarise for.
        start: The position of the first key.
        keys: The leaf's ``(C, d)`` keys.
        values: Their values.

    Returns:
        The leaf tuple.
    """
    size = (reference_queries.shape[0], keys.shape[0])
    compressed, biases = _summarise(
        reference_queries, jnp.broadcast_to(keys, (*size, keys.shape[-1])), jnp.zeros(size)
    )
    return start, keys, values, compressed, biases


def subtree_summary(
    reference_queries: jax.Array, value: Leaf, left: Tree[Leaf], right: Tree[Leaf]
) -> tuple[jax.Array, jax.Array]:
    """The ``summarise`` function for `insert` and `map_tree`.

    Merges a node's leaf with its subtrees' summaries, for example
    ``insert(tree, leaf(refs, start, keys, values), partial(subtree_summary, refs))``.

    Args:
        reference_queries: ``(R, d)``, the query directions to summarise for.
        value: The node's own leaf.
        left: The node's left subtree.
        right: The node's right subtree.

    Returns:
        ``(R, d)`` compressed keys and ``(R,)`` biases for the whole subtree.
    """
    summaries = [_summary(c) for c in (left, value, right) if not isinstance(c, Empty)]
    return _summarise(
        reference_queries,
        jnp.stack([k for k, _ in summaries], axis=1),
        jnp.stack([b for _, b in summaries], axis=1),
    )


def build(
    reference_queries: jax.Array, keys: jax.Array, values: jax.Array, chunk_size: int
) -> Tree[Leaf]:
    """Index every full leaf of ``chunk_size`` keys after the sink.

    Args:
        reference_queries: ``(R, d)``, the query directions to summarise for.
        keys: The ``(T, d)`` key cache.
        values: The ``(T, d)`` value cache.
        chunk_size: ``C``, the number of keys per leaf.

    Returns:
        A tree of ``(T - 1) // C`` leaves. Keys after the last full leaf stay out.
    """
    starts = range(1, keys.shape[0] - chunk_size + 1, chunk_size)
    leaves = (
        leaf(reference_queries, s, keys[s : s + chunk_size], values[s : s + chunk_size])
        for s in starts
    )
    summarise = partial(subtree_summary, reference_queries)
    initial: Tree[Leaf] = EMPTY
    return reduce(lambda tree, value: insert(tree, value, summarise), leaves, initial)


def resummarise(tree: Tree[Leaf], reference_queries: jax.Array) -> Tree[Leaf]:
    """The same tree, summarised for new reference queries.

    Each node is summarised once, bottom-up, with `map_tree`, rather than
    replaying every insertion. This is the step to differentiate in training.

    Args:
        tree: A tree from `build`.
        reference_queries: ``(R, d)``, the new query directions.

    Returns:
        A tree with the same shape and leaves, and new summaries.
    """
    return map_tree(
        tree,
        lambda value: leaf(reference_queries, value[0], value[1], value[2]),
        partial(subtree_summary, reference_queries),
    )


def score(chunk: Chunk, query: jax.Array) -> jax.Array:
    """Estimate the log attention mass ``query`` puts on ``chunk``'s keys.

    The best of the chunk's lines, one per reference query. The true log
    mass is convex in ``query`` and each line touches it at its reference
    query, so the estimate is never too high, and exact at a reference query.

    Args:
        chunk: A leaf, or a subtree of leaves.
        query: A ``(d,)`` query, or ``(Q, d)`` queries.

    Returns:
        The estimated log of the sum of ``exp(logit(query, key))`` over the
        chunk: a scalar, or ``(Q,)``.
    """
    return _scores((chunk,), query)[..., 0]


def _scores(chunks: tuple[Chunk, ...], queries: jax.Array) -> jax.Array:
    """`score` for many chunks in one step: ``(..., n)`` for ``(..., d)`` queries."""
    summaries = [_summary(c) for c in chunks]
    keys = jnp.stack([k for k, _ in summaries])
    logits = jnp.einsum("...d,nrd->...nr", queries, keys) / math.sqrt(queries.shape[-1])
    return jnp.max(logits + jnp.stack([b for _, b in summaries]), axis=-1)


def parts(chunk: Chunk) -> tuple[Chunk, ...]:
    """Split a chunk into the candidates that replace it when it is opened.

    Args:
        chunk: A leaf, or a subtree of leaves.

    Returns:
        A subtree's left subtree, own leaf and right subtree, leaving out
        empty sides. A leaf on its own.
    """
    match chunk:
        case Node(value, left, right):
            return tuple(c for c in (left, value, right) if not isinstance(c, Empty))
        case _:
            return (chunk,)


def _descend(
    tree: Tree[Leaf], pick: Callable[[tuple[Chunk, ...]], tuple[Chunk, ...]]
) -> tuple[tuple[Chunk, ...], ...]:
    """Call ``pick`` on each round's candidates, from the root down, until only leaves are left.

    Each round's candidates are the parts of the chunks picked the round before.
    """

    def rounds(candidates: tuple[Chunk, ...]) -> tuple[tuple[Chunk, ...], ...]:
        picked = pick(candidates)
        if not any(isinstance(c, Node) for c in candidates):
            return (picked,)
        return (picked, *rounds(tuple(part for c in picked for part in parts(c))))

    return rounds(() if isinstance(tree, Empty) else (tree,))


def route(tree: Tree[Leaf], query: jax.Array, p: float) -> tuple[Chunk, ...]:
    """Search the tree for the leaves ``query`` should read, most promising first.

    The search keeps a set of open candidates, starting with the root. Each
    step takes the highest-scoring one: a leaf is picked, and a subtree is
    opened, replaced by its `parts`. It stops once the picked leaves hold
    ``p`` of the estimated mass of everything picked or still open. A single
    threshold covers the whole search, so a deep tree loses nothing per level.

    Scores are computed with NumPy, not JAX: routing needs no gradients, and
    each step scores only a few chunks, where JAX's per-call cost dominates.

    Args:
        tree: A tree from `build`.
        query: The ``(d,)`` query.
        p: The share of estimated attention mass to pick, in (0, 1].

    Returns:
        Every chunk the search took, in order: the subtrees it opened and the
        leaves it picked. Attention reads the leaves.
    """
    q = np.asarray(query, dtype=np.float64)

    def scored(chunks: tuple[Chunk, ...]) -> tuple[tuple[float, Chunk], ...]:
        summaries = [_summary(c) for c in chunks]
        lines = [np.asarray(k) @ q / math.sqrt(q.shape[-1]) + np.asarray(b) for k, b in summaries]
        return tuple((float(np.max(line)), c) for line, c in zip(lines, chunks, strict=True))

    type State = tuple[tuple[tuple[float, Chunk], ...], tuple[tuple[float, Chunk], ...]]

    def step(state: State) -> State:
        candidates, taken = state
        best = max(candidates, key=lambda item: item[0])
        rest = tuple(item for item in candidates if item is not best)
        opened = scored(parts(best[1])) if isinstance(best[1], Node) else ()
        return (*rest, *opened), (*taken, best)

    def done(state: State) -> bool:
        candidates, taken = state
        picked = [s for s, c in taken if not isinstance(c, Node)]
        scores = np.array([s for s, _ in candidates] + picked)
        weights = np.exp(scores - scores.max()) if len(scores) else scores
        return not candidates or weights[len(candidates) :].sum() >= p * weights.sum()

    start: State = (() if isinstance(tree, Empty) else scored((tree,)), ())
    states = accumulate(repeat(None), lambda state, _: step(state), initial=start)
    _, taken = next(state for state in states if done(state))
    return tuple(chunk for _, chunk in taken)


def attend(
    tree: Tree[Leaf], keys: jax.Array, values: jax.Array, query: jax.Array, p: float
) -> jax.Array:
    """Attend over the sink, the routed leaves, and the keys after the last full leaf.

    ``p=1.0`` keeps every leaf, so it gives ordinary dense attention.

    Args:
        tree: A tree from `build` over ``keys`` and ``values``.
        keys: The ``(T, d)`` key cache.
        values: The ``(T, d)`` value cache.
        query: The ``(d,)`` query.
        p: The share of estimated attention mass to pick, in (0, 1].

    Returns:
        The softmax-weighted average of the selected values.
    """
    leaves = [c for c in route(tree, query, p) if not isinstance(c, Node)]
    match tree:
        case Node():
            last_start, last_keys, *_ = maximum(tree)
            tail = last_start + last_keys.shape[0]
        case Empty():
            tail = 1
    selected_keys = jnp.concatenate([keys[:1], *(k for _, k, *_ in leaves), keys[tail:]])
    selected_values = jnp.concatenate([values[:1], *(v for _, _, v, *_ in leaves), values[tail:]])
    return jax.nn.softmax(_logits(query, selected_keys)) @ selected_values


def _log_masses(tree: Tree[Leaf], queries: jax.Array) -> dict[int, jax.Array]:
    """The true log attention mass ``(Q, d)`` queries put on every leaf and subtree, by `id`."""
    leaves = list(in_order(tree))
    if not leaves:
        return {}
    logits = jnp.einsum("qd,ncd->qnc", queries, jnp.stack([k for _, k, *_ in leaves]))
    leaf_masses = logsumexp(logits / math.sqrt(queries.shape[-1]), axis=-1)
    by_leaf = {id(value): leaf_masses[:, i] for i, value in enumerate(leaves)}

    def subtrees(t: Tree[Leaf]) -> dict[int, jax.Array]:
        match t:
            case Node(value, left, right):
                below = subtrees(left) | subtrees(right)
                children = [below[id(c)] for c in (left, right) if isinstance(c, Node)]
                own = by_leaf[id(value)]
                return below | {id(t): logsumexp(jnp.stack([*children, own]), axis=0)}
            case Empty():
                return {}

    return by_leaf | subtrees(tree)


def loss(tree: Tree[Leaf], queries: jax.Array) -> jax.Array:
    """The training loss: how far the scores' chunk shares are from the true ones.

    Opening every subtree, one level per round, splits the leaves into
    chunks in a new way each round. For each round, the true shares are
    dense attention summed per chunk and
    the predicted shares are a softmax over `score`. The loss is the KL
    divergence between them, summed over rounds and averaged over queries.
    The sink and the keys after the last leaf are always attended, so they
    are left out.

    Args:
        tree: A tree from `build` or `resummarise`.
        queries: ``(Q, d)`` queries, for example from the frozen model.

    Returns:
        The scalar loss.
    """
    masses = _log_masses(tree, queries)

    def round_kl(chunks: tuple[Chunk, ...]) -> jax.Array:
        true = jax.nn.log_softmax(jnp.stack([masses[id(c)] for c in chunks], axis=-1))
        pred = jax.nn.log_softmax(_scores(chunks, queries))
        return jnp.sum(jnp.exp(true) * (true - pred), axis=-1)

    rounds = [c for c in _descend(tree, lambda chunks: chunks) if c]
    return jnp.mean(sum((round_kl(c) for c in rounds), start=jnp.zeros(queries.shape[0])))


def main() -> None:
    """Build a tree over a random cache and compare one query's ZHA output with dense attention."""
    keys, values, refs, query = (
        jax.random.normal(jax.random.key(seed), shape)
        for seed, shape in enumerate([(1025, 16), (1025, 16), (8, 16), (16,)])
    )
    tree = build(refs, keys, values, chunk_size=16)
    dense = jax.nn.softmax(_logits(query, keys)) @ values
    for p in (0.5, 0.9, 1.0):
        leaves = [c for c in route(tree, query, p) if not isinstance(c, Node)]
        error = float(jnp.abs(attend(tree, keys, values, query, p) - dense).max())
        print(f"p={p}: reads {len(leaves)} of {size(tree)} leaves, max error {error:.4f}")  # noqa: T201


if __name__ == "__main__":
    main()
