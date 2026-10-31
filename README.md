# safe_tree

A purely functional, immutable, self-balancing (AVL) tree for Python.
Every operation returns a new tree; old versions stay valid.

```python
from safe_tree import EMPTY, insert, delete, member, in_order

v1 = insert(insert(insert(EMPTY, 5), 2), 8)
v2 = insert(v1, 7)
v3 = delete(v2, 2)

list(in_order(v1))  # [2, 5, 8]       <- v1 never changes
list(in_order(v2))  # [2, 5, 7, 8]
list(in_order(v3))  # [5, 7, 8]
member(v1, 7)  # False — v1 is exactly what it was before v2 existed
```

## Install

Not yet published to PyPI. With [uv](https://docs.astral.sh/uv/):

```sh
uv add git+https://github.com/mateodiazs/safe_tree
```

## The shape of a tree

```
Tree[A]
├── Empty                 — exactly one instance, the module-level EMPTY
└── Node[A]
    ├── value  : A
    ├── left   : Tree[A]
    ├── right  : Tree[A]
    ├── height : int        — cached, kept correct by insert/delete
    └── size   : int        — cached, kept correct by insert/delete
```

Build nodes via `insert`/`EMPTY`/`from_iterable`, not `Node(...)`
directly with non-empty children, so `height`/`size` stay correct.
`height`/`size` are excluded from `Node.__match_args__`, so
`case Node(value, left, right):` still works.

## Immutability

```python
from safe_tree import EMPTY, Node, insert

node = insert(EMPTY, 5)
assert isinstance(node, Node)
node.value = 999
# mypy:    error: Property "value" defined in "Node" is read-only  [misc]
# runtime: dataclasses.FrozenInstanceError: cannot assign to field 'value'
```

Two limits, and neither is specific to this library. `object.__setattr__`
bypasses `frozen=True` on any Python dataclass. A mutable *value* stored
in a tree can still be mutated in place, so store immutable values.

## Structural sharing

```
v1 = insert(insert(insert(EMPTY, 5), 2), 8)

        5
       / \
      2   8

v2 = insert(v1, 7)

        5'
       /  \
      2    8'
            \
             7
```

```python
tree = insert(insert(insert(EMPTY, 5), 2), 8)
updated = insert(tree, 7)
assert tree.left is updated.left  # the same object, not a copy
```

`tests/test_sharing.py` asserts this with `is`, not `==`. When a
rotation fires, a couple of extra nodes get rebuilt beyond the bare
insertion path. Still O(log n), just not always the minimum.

## Staying balanced

```python
from safe_tree import EMPTY, insert, depth, size

tree = EMPTY
for i in range(100_000):
    tree = insert(tree, i)  # worst case for an unbalanced BST

size(tree)  # 100000
depth(tree)  # 17 — not 100000, and not a RecursionError
```

## Equality

```python
by_ascending = insert(insert(insert(insert(EMPTY, 0), 1), 2), 3)
by_other_order = insert(insert(insert(insert(EMPTY, 0), 1), 3), 2)

list(in_order(by_ascending)) == list(in_order(by_other_order))  # True  — same values
by_ascending == by_other_order  # False — different (both valid) AVL shapes

from safe_tree import equal

equal(by_ascending, by_other_order)  # True — compares values, ignores shape
```

## A slightly bigger example

```python
from safe_tree import EMPTY, fold, filter_tree, map_tree, insert, in_order

scores = [55, 72, 88, 91, 63, 79, 45]

tree = EMPTY
for score in scores:
    tree = insert(tree, score)

passing = filter_tree(tree, lambda s: s >= 60)
curved = map_tree(passing, lambda s: min(100, s + 5))
total = fold(curved, lambda acc, s: acc + s, 0)

list(in_order(curved))  # [68, 77, 84, 93, 96]
round(total / len(list(in_order(curved))), 1)  # 83.6
```

`tree`, `passing`, and `curved` are three independent trees.

## Order statistics and range queries

```python
from safe_tree import minimum, maximum, kth, range_query

minimum(tree)  # 45                — O(log n)
maximum(tree)  # 91                — O(log n)
kth(tree, 0)  # 45 — smallest       — O(log n), via cached size
kth(tree, 3)  # 72 — 4th smallest
list(range_query(tree, 60, 90))  # [63, 72, 79, 88] — O(log n + k), pruned
```

## Set algebra

```python
from safe_tree import EMPTY, insert, in_order, union, intersection, difference

a, b = EMPTY, EMPTY
for v in [1, 2, 3, 4]:
    a = insert(a, v)
for v in [3, 4, 5, 6]:
    b = insert(b, v)

list(in_order(union(a, b)))  # [1, 2, 3, 4, 5, 6]
list(in_order(intersection(a, b)))  # [3, 4]
list(in_order(difference(a, b)))  # [1, 2]
```

Built on `insert`/`member`/`filter_tree` directly, so it's
O(size(other) * log n) rather than the optimal O(m + n) merge.

## Lenses

```
Node[Point]  --value_lens()-->  Point  --x_lens-->  int
     \_____________ then(value_lens(), x_lens) _____________/
```

```python
from dataclasses import dataclass
from safe_tree import EMPTY, Lens, Node, then, value_lens


@dataclass(frozen=True, slots=True)
class Point:
    x: int
    y: int


x_lens: Lens[Point, int] = Lens(
    get=lambda p: p.x,  # read x out of a Point
    put=lambda p, x: Point(x, p.y),  # new Point with x swapped in, same y
)

node = Node(Point(1, 2), EMPTY, EMPTY)
node_x = then(value_lens(), x_lens)  # Node -> its value -> that value's x

node_x.get(node)  # 1
node_x.put(node, 9).value  # Point(x=9, y=2) — node itself is untouched
```

`value_lens()`, not `value_lens` bare. These are generic functions, not
pre-built values, so `mypy` can resolve the type parameter.

No general `path_lens`. A node's own fields always exist and are safe
to get/put, but a deeper tree position might not. That's a job for
pattern matching, not a lens.

## API

| | |
|---|---|
| `insert(tree, value)` | New tree with `value` added, self-balancing. |
| `delete(tree, value)` | New tree with `value` removed, self-balancing. |
| `member(tree, value)` | Membership test. |
| `equal(a, b)` | Same values, regardless of shape (`==` also works, but is shape-sensitive). |
| `minimum(tree)` / `maximum(tree)` | Smallest / largest value, O(log n). |
| `kth(tree, k)` | The k-th smallest value (0-indexed), O(log n). |
| `range_query(tree, lo, hi)` | Values in `[lo, hi]`, O(log n + k). |
| `union` / `intersection` / `difference` | Set algebra between two trees. |
| `from_iterable(values)` | Build a tree from any iterable. |
| `map_tree(tree, f)` | Structure-preserving map over every value. |
| `fold(tree, f, init)` | In-order reduction to a single value. |
| `filter_tree(tree, pred)` | Rebuild keeping only matching values. |
| `in_order` / `pre_order` / `post_order` | Lazy traversal iterators. |
| `size(tree)` / `depth(tree)` | Structural measures, both O(1). |
| `Lens`, `then`, `value_lens`, `left_lens`, `right_lens` | Composable node accessors. |

```sh
uv run pdoc -o docs safe_tree  # full API reference from the same docstrings
```

## Playground

A local web app with two independent trees. Insert/delete with a
version timeline (chips marked **rotated** when AVL actually fires),
min/max/kth, a `range_query` slider, and a Combine panel for
union/intersection/difference.

```sh
uv run --group playground uvicorn playground.app:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`, or `http://<your-LAN-IP>:8000` from a
phone on the same Wi-Fi, or over Tailscale from anywhere.

## Development

```sh
uv sync --all-extras --all-groups
uv run ruff check .                           # lint
uv run ruff format .                          # format
uv run mypy src tests playground conftest.py  # strict type-check
uv run pytest                                 # tests, incl. doctests, hypothesis properties, and JAX (skipped if not installed)
```

## JAX

```python
import jax
from safe_tree import EMPTY, insert, in_order
from safe_tree.jax_support import register

register()  # once, after importing jax; jax is optional (uv sync --group jax)

tree = insert(insert(insert(EMPTY, 5), 2), 8)
jax.tree_util.tree_leaves(tree)  # [5, 2, 8]
list(in_order(jax.tree_util.tree_map(lambda v: v * 2, tree)))  # [4, 10, 16]
```
