from collections.abc import Sequence

from hypothesis import given
from hypothesis import strategies as st

from safe_tree import EMPTY, Tree, delete, fold, in_order, insert, map_tree, member, size


def build(values: Sequence[int]) -> Tree[int]:
    tree: Tree[int] = EMPTY
    for value in values:
        tree = insert(tree, value)
    return tree


@given(st.lists(st.integers()))
def test_every_inserted_value_is_a_member(values: list[int]) -> None:
    tree = build(values)
    assert all(member(tree, v) for v in values)


@given(st.lists(st.integers(), min_size=1))
def test_delete_removes_a_value(values: list[int]) -> None:
    tree = build(values)
    target = values[0]
    without = delete(tree, target)
    if values.count(target) == 0:
        assert without is tree
    else:
        assert not member(without, target)


@given(st.lists(st.integers()))
def test_in_order_is_sorted_and_deduplicated(values: list[int]) -> None:
    tree = build(values)
    assert list(in_order(tree)) == sorted(set(values))


@given(st.lists(st.integers()))
def test_size_matches_unique_count(values: list[int]) -> None:
    tree = build(values)
    assert size(tree) == len(set(values))


@given(st.lists(st.integers()))
def test_fold_sum_matches_python_sum(values: list[int]) -> None:
    tree = build(values)
    assert fold(tree, lambda acc, v: acc + v, 0) == sum(set(values))


@given(st.lists(st.integers()))
def test_map_tree_preserves_shape(values: list[int]) -> None:
    tree = build(values)
    doubled = map_tree(tree, lambda v: v * 2)
    assert list(in_order(doubled)) == [v * 2 for v in sorted(set(values))]
