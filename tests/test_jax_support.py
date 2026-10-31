import pytest

jax = pytest.importorskip("jax")

from safe_tree import EMPTY, Tree, in_order, insert  # noqa: E402
from safe_tree.jax_support import register  # noqa: E402

register()
register()  # idempotency check


def _build(values: list[int]) -> Tree[int]:
    tree: Tree[int] = EMPTY
    for value in values:
        tree = insert(tree, value)
    return tree


def test_tree_leaves_are_its_values() -> None:
    tree = _build([5, 2, 8])
    assert sorted(jax.tree_util.tree_leaves(tree)) == [2, 5, 8]


def test_empty_has_no_leaves() -> None:
    assert jax.tree_util.tree_leaves(EMPTY) == []


def test_tree_map_matches_map_tree() -> None:
    tree = _build([5, 2, 8, 1, 9])
    doubled = jax.tree_util.tree_map(lambda v: v * 2, tree)
    assert list(in_order(doubled)) == [v * 2 for v in [1, 2, 5, 8, 9]]
