import math

import pytest

pytest.importorskip("jax")

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from safe_tree.attention.zha import attend, build, loss, route, scores

D, CHUNK, DEPTH = 8, 4, 3
T = 2 * CHUNK**DEPTH
TOL = 1e-4


def _normal(seed: int, *shape: int) -> jax.Array:
    return jax.random.normal(jax.random.key(seed), shape)


KEYS, VALUES, REFS = _normal(1, T, D), _normal(2, T, D), _normal(3, 5, D)
LEVELS = build(REFS, KEYS, CHUNK, DEPTH)


def _true_mass(keys: jax.Array, depth: int, query: jax.Array) -> jax.Array:
    """Log attention mass ``query`` puts on each node ``depth + 1`` levels up, from the raw keys."""
    return logsumexp((keys @ query / math.sqrt(D)).reshape(-1, CHUNK ** (depth + 1)), axis=-1)


def _every_node(level: tuple[jax.Array, jax.Array]) -> jax.Array:
    return jnp.arange(level[1].shape[1])


def test_each_level_has_chunk_times_fewer_nodes() -> None:
    assert [level[1].shape for level in LEVELS] == [(5, 32), (5, 8), (5, 2)]


def test_scores_are_exact_at_each_reference_query() -> None:
    assert all(
        jnp.allclose(scores(level, q, _every_node(level)), _true_mass(KEYS, i, q), atol=TOL)
        for i, level in enumerate(LEVELS)
        for q in REFS
    )


def test_scores_never_exceed_the_true_log_mass() -> None:
    assert all(
        jnp.all(scores(level, q, _every_node(level)) <= _true_mass(KEYS, i, q) + TOL)
        for i, level in enumerate(LEVELS)
        for q in _normal(4, 50, D)
    )


def test_route_follows_the_scores_down_to_a_planted_key() -> None:
    query, needle = _normal(5, D), 77
    keys = KEYS.at[needle].set(4 * query)
    levels = build(jnp.concatenate([REFS, query[None]]), keys, CHUNK, DEPTH)
    assert needle // CHUNK in route(levels, query, k=1, chunk=CHUNK).tolist()


def test_route_keeps_k_leaves_even_below_a_single_top_node() -> None:
    levels = build(REFS, KEYS[: CHUNK**DEPTH], CHUNK, DEPTH)
    leaves = route(levels, _normal(10, D), k=3, chunk=CHUNK).tolist()
    assert levels[-1][1].shape[1] == 1
    assert len(set(leaves)) == 3


def test_attend_is_dense_attention_when_k_covers_every_leaf() -> None:
    levels, query = build(REFS, KEYS, CHUNK, 1), _normal(6, D)
    dense = jax.nn.softmax(KEYS @ query / math.sqrt(D)) @ VALUES
    sparse = attend(levels, KEYS, VALUES, query, k=T // CHUNK)
    assert jnp.allclose(sparse, dense, atol=TOL)


def test_attend_batches_under_jit_and_vmap() -> None:
    queries = _normal(7, 16, D)
    batched = jax.jit(jax.vmap(attend, in_axes=(None, None, None, 0, None)), static_argnums=4)
    one_by_one = jnp.stack([attend(LEVELS, KEYS, VALUES, q, 2) for q in queries])
    assert jnp.allclose(batched(LEVELS, KEYS, VALUES, queries, 2), one_by_one, atol=TOL)


def test_loss_is_zero_at_the_reference_queries() -> None:
    assert float(loss(REFS, KEYS, REFS, CHUNK, DEPTH)) < TOL


def test_gradient_steps_lower_the_loss() -> None:
    queries = 2 * (0.3 * _normal(8, 32, D) + _normal(9, D))
    gradient = jax.jit(jax.grad(loss), static_argnums=(3, 4))
    refs = REFS
    for _ in range(10):
        refs = refs - gradient(refs, KEYS, queries, CHUNK, DEPTH)
    before, after = (float(loss(r, KEYS, queries, CHUNK, DEPTH)) for r in (REFS, refs))
    assert after < 0.6 * before
