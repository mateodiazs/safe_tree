import dataclasses
import math
from collections.abc import Sequence

import pytest
from hypothesis import given
from hypothesis import strategies as st

from safe_tree import (
    EMPTY,
    Node,
    Tree,
    delete,
    depth,
    difference,
    equal,
    fold,
    from_iterable,
    in_order,
    insert,
    intersection,
    kth,
    map_tree,
    maximum,
    member,
    minimum,
    range_query,
    size,
    union,
)


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


@st.composite
def _values_and_a_shuffle(draw: st.DrawFn) -> tuple[list[int], list[int]]:
    values = draw(st.lists(st.integers()))
    return values, draw(st.permutations(values))


@given(_values_and_a_shuffle())
def test_equal_ignores_insertion_order(pair: tuple[list[int], list[int]]) -> None:
    values, shuffled = pair
    assert equal(build(values), build(shuffled))


@given(st.lists(st.integers()), st.integers())
def test_equal_detects_a_different_value(values: list[int], extra: int) -> None:
    tree = build(values)
    with_extra = insert(tree, extra)

    assert equal(tree, with_extra) == (extra in set(values))


def test_dataclass_eq_is_shape_sensitive_but_equal_is_not() -> None:
    # Same four values, two insertion orders that settle into different
    # (both individually valid) AVL shapes.
    by_ascending = build([0, 1, 2, 3])
    by_other_order = build([0, 1, 3, 2])

    assert list(in_order(by_ascending)) == list(in_order(by_other_order))
    assert by_ascending != by_other_order  # different shape
    assert equal(by_ascending, by_other_order)  # same values


@given(st.lists(st.integers()))
def test_tree_stays_balanced(values: list[int]) -> None:
    tree = build(values)
    n = size(tree)
    if n == 0:
        assert depth(tree) == 0
        return
    # True AVL worst case is ~1.44 * log2(n); 2x log2 leaves comfortable margin
    # without pinning the test to the exact constant.
    assert depth(tree) <= 2 * math.log2(n + 1)


def test_tree_stays_balanced_after_ten_thousand_sorted_inserts() -> None:
    # The scenario that crashes a plain, unbalanced BST with a
    # RecursionError around depth 1000 (Python's recursion limit).
    tree = build(range(10_000))
    assert size(tree) == 10_000
    assert depth(tree) <= 2 * math.log2(10_001)


@given(st.lists(st.integers(), min_size=1))
def test_minimum_and_maximum(values: list[int]) -> None:
    tree = build(values)
    assert minimum(tree) == min(values)
    assert maximum(tree) == max(values)


def test_minimum_of_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        minimum(EMPTY)


def test_maximum_of_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        maximum(EMPTY)


@given(st.lists(st.integers(), min_size=1))
def test_kth_matches_sorted_index(values: list[int]) -> None:
    tree = build(values)
    for k, expected in enumerate(sorted(set(values))):
        assert kth(tree, k) == expected


def test_kth_out_of_range_raises() -> None:
    tree = build([1, 2, 3])
    with pytest.raises(IndexError):
        kth(tree, 3)


@given(st.lists(st.integers()), st.integers(), st.integers())
def test_range_query_matches_a_filtered_sort(values: list[int], lo: int, hi: int) -> None:
    tree = build(values)
    expected = sorted(v for v in set(values) if lo <= v <= hi)
    assert list(range_query(tree, lo, hi)) == expected


@given(st.lists(st.integers()))
def test_from_iterable_matches_sequential_insert(values: list[int]) -> None:
    assert list(in_order(from_iterable(values))) == list(in_order(build(values)))


@given(st.lists(st.integers()), st.lists(st.integers()))
def test_union_matches_set_union(a: list[int], b: list[int]) -> None:
    result = union(build(a), build(b))
    assert set(in_order(result)) == set(a) | set(b)


@given(st.lists(st.integers()), st.lists(st.integers()))
def test_intersection_matches_set_intersection(a: list[int], b: list[int]) -> None:
    result = intersection(build(a), build(b))
    assert set(in_order(result)) == set(a) & set(b)


@given(st.lists(st.integers()), st.lists(st.integers()))
def test_difference_matches_set_difference(a: list[int], b: list[int]) -> None:
    result = difference(build(a), build(b))
    assert set(in_order(result)) == set(a) - set(b)


def test_node_fields_cannot_be_reassigned() -> None:
    node = insert(EMPTY, 5)
    assert isinstance(node, Node)
    with pytest.raises(dataclasses.FrozenInstanceError):
        node.value = 999  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        node.left = node  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        node.right = node  # type: ignore[misc]


def test_empty_has_no_settable_attributes() -> None:
    with pytest.raises(AttributeError):
        EMPTY.value = 5  # type: ignore[attr-defined]
