from dataclasses import dataclass

from hypothesis import given
from hypothesis import strategies as st

from safe_tree import EMPTY, Lens, Node, Tree, left_lens, right_lens, then, value_lens


def leaf(value: int) -> Node[int]:
    return Node(value, EMPTY, EMPTY)


@given(st.integers(), st.integers())
def test_value_lens_get_put_law(initial: int, new_value: int) -> None:
    node = leaf(initial)
    lens: Lens[Node[int], int] = value_lens()

    assert lens.get(node) == initial
    assert lens.put(node, new_value) == Node(new_value, EMPTY, EMPTY)


@given(st.integers())
def test_value_lens_put_get_is_identity(value: int) -> None:
    node = leaf(0)
    lens: Lens[Node[int], int] = value_lens()

    updated = lens.put(node, value)
    assert lens.get(updated) == value


@given(st.integers())
def test_left_lens_round_trips(value: int) -> None:
    node = leaf(0)
    child: Node[int] = leaf(value)
    lens: Lens[Node[int], Tree[int]] = left_lens()

    updated = lens.put(node, child)
    assert lens.get(updated) is child


@given(st.integers())
def test_right_lens_round_trips(value: int) -> None:
    node = leaf(0)
    child: Node[int] = leaf(value)
    lens: Lens[Node[int], Tree[int]] = right_lens()

    updated = lens.put(node, child)
    assert lens.get(updated) is child


@dataclass(frozen=True, slots=True)
class Point:
    x: int
    y: int


x_lens: Lens[Point, int] = Lens(
    get=lambda point: point.x,
    put=lambda point, x: Point(x, point.y),
)


def point_leaf(point: Point) -> Node[Point]:
    return Node(point, EMPTY, EMPTY)


@given(st.integers(), st.integers(), st.integers())
def test_composed_lens_reaches_into_a_node_payload(x: int, y: int, new_x: int) -> None:
    node = point_leaf(Point(x, y))
    node_x_lens: Lens[Node[Point], int] = then(value_lens(), x_lens)
    node_value_lens: Lens[Node[Point], Point] = value_lens()

    updated = node_x_lens.put(node, new_x)

    assert node_x_lens.get(node) == x
    assert node_value_lens.get(updated) == Point(new_x, y)
