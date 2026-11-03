"""Zyphra Hierarchical Attention (ZHA) in JAX: a tree of key summaries over a KV cache.

From "Subquadratic Attention with a Hierarchical Tree-Structured Indexer"
(Lufkin et al., arXiv:2610.42112). Every ``chunk`` neighbouring keys are
summarised into one node, every ``chunk`` nodes into a parent, and so on
for ``depth`` levels. A node stores one compressed key and one bias per
reference query; its score for a query is the best of those lines, an
estimate of the log attention mass the query puts on the node's keys,
never too high and exact at a reference query. A query keeps the top
``k`` nodes at each level from the top down, then attends densely to the
keys in the ``k`` leaves it reached.

Each level is a pair of arrays, so every function here works under
`jax.jit`, `jax.vmap` and `jax.grad`. The reference queries are the only
learned parameters: `loss` is the paper's training objective.

Not covered: the RoPE correction, and the paper's top-p selection, which
this replaces with a fixed top-``k`` per level so shapes stay static.
Attention covers only the routed keys; the paper also always reads the
first key and the newest keys, which a caller can attend to separately.

Needs the ``jax`` dependency group (``uv sync --group jax``).
"""

from __future__ import annotations

import math
from functools import reduce
from itertools import accumulate

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

type Level = tuple[jax.Array, jax.Array]
"""One level of the tree: ``(R, N, d)`` compressed keys and ``(R, N)`` biases,
one row per reference query, one column per node."""


def merge(refs: jax.Array, level: Level, chunk: int) -> Level:
    """Summarise every ``chunk`` neighbouring nodes of ``level`` into one parent.

    Row ``r`` of a parent's compressed keys is what reference query ``r``
    would read from its children if the values were the keys themselves.
    The bias makes the line ``q @ key / sqrt(d) + bias`` touch the parent's
    log attention mass at that reference query.

    Args:
        refs: ``(R, d)`` reference queries.
        level: The level to summarise, with ``N`` a multiple of ``chunk``.
        chunk: Children per parent.

    Returns:
        The parent level, with ``N // chunk`` nodes.
    """
    keys, biases = level
    r, n, d = keys.shape
    keys, biases = keys.reshape(r, n // chunk, chunk, d), biases.reshape(r, n // chunk, chunk)
    logits = jnp.einsum("rd,rncd->rnc", refs, keys) / math.sqrt(d) + biases
    compressed = jnp.einsum("rnc,rncd->rnd", jax.nn.softmax(logits, axis=-1), keys)
    on_line = jnp.einsum("rd,rnd->rn", refs, compressed) / math.sqrt(d)
    return compressed, logsumexp(logits, axis=-1) - on_line


def build(refs: jax.Array, keys: jax.Array, chunk: int, depth: int) -> tuple[Level, ...]:
    """Summarise ``keys`` into ``depth`` levels, leaves first.

    Args:
        refs: ``(R, d)`` reference queries.
        keys: ``(T, d)`` keys, with ``T`` a multiple of ``chunk ** depth``.
        chunk: Children per node; also the number of keys per leaf.
        depth: The number of levels.

    Returns:
        ``depth`` levels: leaves of ``chunk`` keys first, the top level last.
    """
    shape = (refs.shape[0], keys.shape[0])
    raw: Level = (jnp.broadcast_to(keys, (*shape, keys.shape[1])), jnp.zeros(shape))
    levels = accumulate(range(depth), lambda level, _: merge(refs, level, chunk), initial=raw)
    return tuple(levels)[1:]


def scores(level: Level, query: jax.Array, nodes: jax.Array) -> jax.Array:
    """Estimate the log attention mass ``query`` puts on each of ``nodes``.

    Args:
        level: The level the nodes are on.
        query: A ``(d,)`` query.
        nodes: Indices of nodes on ``level``.

    Returns:
        One score per node: the best of its lines, one per reference query.
    """
    keys, biases = level
    lines = keys[:, nodes] @ query / math.sqrt(query.shape[-1]) + biases[:, nodes]
    return jnp.max(lines, axis=0)


def route(levels: tuple[Level, ...], query: jax.Array, k: int, chunk: int) -> jax.Array:
    """Keep the ``k`` best nodes at each level, from the top down.

    The top level's nodes are the first candidates. At each level below,
    the candidates are the children of the nodes kept above. A level with
    fewer than ``k`` candidates keeps them all.

    Args:
        levels: Levels from `build`.
        query: A ``(d,)`` query.
        k: Nodes kept per level.
        chunk: Children per node, as given to `build`.

    Returns:
        The leaf indices the query reaches: ``k`` of them, or every leaf if there are fewer.
    """
    top = jnp.arange(levels[-1][1].shape[1])

    def keep(candidates: jax.Array, level: Level) -> jax.Array:
        kept = min(k, candidates.shape[0])
        return candidates[jax.lax.top_k(scores(level, query, candidates), kept)[1]]

    def descend(kept: jax.Array, level: Level) -> jax.Array:
        return keep((kept[:, None] * chunk + jnp.arange(chunk)).ravel(), level)

    return reduce(descend, reversed(levels[:-1]), keep(top, levels[-1]))


def attend(
    levels: tuple[Level, ...], keys: jax.Array, values: jax.Array, query: jax.Array, k: int
) -> jax.Array:
    """Attend densely to the keys in the ``k`` leaves `route` reaches.

    For a batch of queries, ``jax.vmap`` over ``query``, keeping ``k``
    static under ``jax.jit``.

    Args:
        levels: Levels from `build` over ``keys``.
        keys: The ``(T, d)`` keys.
        values: The ``(T, d_v)`` values.
        query: A ``(d,)`` query.
        k: Leaves kept per level.

    Returns:
        The ``(d_v,)`` softmax-weighted average of the selected values.
    """
    chunk = keys.shape[0] // levels[0][1].shape[1]
    selected = (route(levels, query, k, chunk)[:, None] * chunk + jnp.arange(chunk)).ravel()
    weights = jax.nn.softmax(keys[selected] @ query / math.sqrt(query.shape[-1]))
    return weights @ values[selected]


def loss(refs: jax.Array, keys: jax.Array, queries: jax.Array, chunk: int, depth: int) -> jax.Array:
    """The training objective for the reference queries.

    At each level, the true shares are dense attention summed per node and
    the predicted shares are a softmax over `scores`. The loss is the KL
    divergence between them, summed over levels and averaged over queries.
    Differentiate it with respect to ``refs``.

    Args:
        refs: ``(R, d)`` reference queries.
        keys: ``(T, d)`` keys, with ``T`` a multiple of ``chunk ** depth``.
        queries: ``(Q, d)`` queries, for example from the frozen model.
        chunk: Children per node.
        depth: The number of levels.

    Returns:
        The scalar loss.
    """
    logits = queries @ keys.T / math.sqrt(keys.shape[-1])

    def level_kl(depth_index: int, level: Level) -> jax.Array:
        grouped = logits.reshape(queries.shape[0], -1, chunk ** (depth_index + 1))
        true = jax.nn.log_softmax(logsumexp(grouped, axis=-1), axis=-1)
        every = jnp.arange(level[1].shape[1])
        pred = jax.nn.log_softmax(jax.vmap(lambda q: scores(level, q, every))(queries), axis=-1)
        return jnp.mean(jnp.sum(jnp.exp(true) * (true - pred), axis=-1))

    levels = build(refs, keys, chunk, depth)
    return sum((level_kl(i, level) for i, level in enumerate(levels)), start=jnp.zeros(()))
