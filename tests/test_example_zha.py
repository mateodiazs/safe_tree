import math
from functools import partial

import pytest

pytest.importorskip("jax")

import jax
import jax.numpy as jnp

from examples.zha import (
    Chunk,
    Leaf,
    attend,
    build,
    leaf,
    loss,
    resummarise,
    route,
    score,
    subtree_summary,
)
from safe_tree import Empty, Node, Tree, depth, in_order, insert, size

DIM = 4
TOL = 1e-4


def _normal(seed: int, *shape: int) -> jax.Array:
    return jax.random.normal(jax.random.key(seed), shape)


def _cache(n: int) -> tuple[jax.Array, jax.Array]:
    return _normal(1, n, DIM), _normal(2, n, DIM)


def _chunks(tree: Tree[Leaf]) -> list[Chunk]:
    match tree:
        case Node(value, left, right):
            return [tree, value, *_chunks(left), *_chunks(right)]
        case Empty():
            return []


def _raw_keys(chunk: Chunk) -> jax.Array:
    match chunk:
        case Node():
            return jnp.concatenate([keys for _, keys, *_ in in_order(chunk)])
        case (_, keys, *_):
            return keys


def _true_mass(chunk: Chunk, query: jax.Array) -> float:
    return float(jax.scipy.special.logsumexp(_raw_keys(chunk) @ query / math.sqrt(DIM)))


REFS = _normal(3, 3, DIM)
TREE = build(REFS, *_cache(38), chunk_size=2)


def test_leaves_cover_every_full_chunk_after_the_sink() -> None:
    assert [start for start, *_ in in_order(TREE)] == list(range(1, 37, 2))


def test_p_one_is_dense_attention() -> None:
    keys, values = _cache(38)
    query = _normal(4, DIM)
    dense = jax.nn.softmax(keys @ query / math.sqrt(DIM)) @ values
    assert jnp.allclose(attend(TREE, keys, values, query, p=1.0), dense, atol=TOL)


def test_score_is_exact_at_each_reference_query() -> None:
    assert depth(TREE) >= 4
    assert all(
        math.isclose(float(score(chunk, q)), _true_mass(chunk, q), abs_tol=TOL)
        for chunk in _chunks(TREE)
        for q in REFS
    )


def test_score_never_exceeds_the_true_log_mass() -> None:
    queries = _normal(5, 50, DIM)
    assert all(float(score(c, q)) <= _true_mass(c, q) + TOL for c in _chunks(TREE) for q in queries)


def test_inserting_a_leaf_shares_all_but_one_path() -> None:
    keys, values = _cache(40)
    grown = insert(TREE, leaf(REFS, 37, keys[37:39], values[37:39]), partial(subtree_summary, REFS))
    assert size(grown) == size(TREE) + 1
    shared = {id(c) for c in _chunks(grown)}
    rebuilt = [c for c in _chunks(TREE) if id(c) not in shared]
    assert all(isinstance(c, Node) for c in rebuilt)
    assert len(rebuilt) <= depth(TREE) + 2


def test_resummarise_matches_building_with_the_new_reference_queries() -> None:
    refs = _normal(6, 3, DIM)
    rebuilt, fresh = resummarise(TREE, refs), build(refs, *_cache(38), chunk_size=2)
    queries = _normal(7, 10, DIM)
    assert all(
        jnp.allclose(score(a, queries), score(b, queries), atol=TOL)
        for a, b in zip(_chunks(rebuilt), _chunks(fresh), strict=True)
    )


def test_small_p_routes_to_fewer_leaves() -> None:
    tree = build(REFS, *_cache(130), chunk_size=4)
    query = _normal(4, DIM)
    leaves = [[c for c in route(tree, query, p) if not isinstance(c, Node)] for p in (0.5, 1.0)]
    assert len(leaves[0]) < len(leaves[1]) == size(tree)


def test_loss_is_zero_when_the_query_is_a_reference_query() -> None:
    assert float(loss(TREE, REFS[:1])) < TOL


def test_gradient_steps_lower_the_loss() -> None:
    queries = (_normal(10, 16, DIM) * 0.3 + _normal(11, DIM)) * 3

    def objective(refs: jax.Array) -> jax.Array:
        return loss(resummarise(TREE, refs), queries)

    refs = REFS
    before = objective(refs)
    for _ in range(5):
        refs = refs - 0.5 * jax.grad(objective)(refs)
    assert float(objective(refs)) < 0.5 * float(before)
