# safe_tree

A purely functional, immutable tree for Python. Every operation returns a
new tree instead of mutating the one you have, so old versions stay valid
forever — and because updates share structure instead of copying it, that
safety doesn't cost you a full copy on every change. It also rebalances
itself (AVL) on every change, so it stays fast and doesn't blow Python's
recursion limit regardless of what order you feed it values in.

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

Three versions of the same tree, all live at once, none of them stepping
on each other. That's the whole pitch.

## Why

- **Immutable by construction.** `insert`, `delete`, `map_tree`, and
  `filter_tree` all return a new tree. Whatever you started with is still
  there, unchanged, for as long as you hold a reference to it.
- **Structural sharing, not copying.** An update only allocates the nodes
  on the path to the change — see [below](#structural-sharing).
- **Self-balancing.** Every `insert`/`delete` rebalances (AVL) on the way
  back up, so depth stays within a constant factor of log2(size) no
  matter what order values arrive in — see [below](#staying-balanced).
- **No behavior hides in classes.** Data lives in frozen structs
  (`Node`, `Lens`); every operation is a plain function over that data.
- **Typed end to end.** `mypy --strict` clean, no implicit `Any`.

## Install

Not yet published to PyPI. With [uv](https://docs.astral.sh/uv/):

```sh
uv add git+https://github.com/mateodiazs/safe_tree
```

## The shape of a tree

A `Tree[A]` is either `Empty`, or a `Node[A]` holding one value and two
child trees:

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

`EMPTY` is never mutated or recreated, so emptiness is always the same
object — that's what makes `delete(tree, missing_value) is tree` true
rather than merely equal. `height`/`size` are bookkeeping for AVL
rebalancing and O(1) `depth`/`size`/`kth` — pattern-matching on a
node's logical shape (`case Node(value, left, right):`) still works,
since they're excluded from `Node.__match_args__`. Construct nodes via
`insert`/`EMPTY`/`from_iterable`, not `Node(...)` directly with
non-empty children, so this bookkeeping stays correct.

## Immutability, concretely

`Node` is `@dataclass(frozen=True, slots=True)`, so reassigning a
field after construction raises — this isn't a convention, it's
enforced, both at runtime and (since mypy treats frozen dataclass
fields as read-only) at type-check time too:

```python
from safe_tree import EMPTY, Node, insert

node = insert(EMPTY, 5)
assert isinstance(node, Node)
node.value = 999
# mypy:    error: Property "value" defined in "Node" is read-only  [misc]
# runtime: dataclasses.FrozenInstanceError: cannot assign to field 'value'
```

`tests/test_tree.py` pins both down. Two honest limits, worth knowing
about rather than assuming away:

- **`object.__setattr__` bypasses `frozen=True`.** This is true of
  *every* frozen dataclass in Python — it's literally how `__init__`
  sets fields on a frozen instance in the first place — not something
  specific to this library, and not fixable without a compiled
  extension type. If code is reaching for `object.__setattr__` to
  mutate a node, it's working against the type on purpose.
- **A mutable *value* stored in a tree can still be mutated in
  place.** `insert(EMPTY, my_list)` keeps `my_list` immutable from the
  tree's perspective — `tree.value is my_list`, not a copy — but
  nothing stops `my_list.append(x)` afterward. Every persistent data
  structure in every language has this same caveat for container
  element types (Haskell's `Data.Map`, Clojure's persistent vectors,
  Java's immutable collections all document it rather than pretend to
  solve it) — the fix is the same here as everywhere else: store
  immutable values.

## Structural sharing

`insert` and `delete` only copy the nodes on the path to the change.
Starting from a tree rooted at `5`, inserting `7` never has a reason to
touch the left subtree at all:

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

Node `2` isn't redrawn in `v2` by accident — it's the exact same Python
object as node `2` in `v1`. `insert` only had to build `5'` and `8'`,
the two nodes on the path down to where `7` belongs:

```python
tree = insert(insert(insert(EMPTY, 5), 2), 8)
updated = insert(tree, 7)
assert tree.left is updated.left  # the same object, not a copy
```

`tests/test_sharing.py` asserts exactly this with `is`, not `==` — the
guarantee is about identity, not just resulting value. The same holds
for `delete`: removing a value that isn't present returns the original
tree unchanged, by identity, not an equal-but-freshly-built one.

Honest caveat: when a rotation *does* fire (see below), it rebuilds a
couple of extra nodes beyond the bare insertion path, since a rotation
changes which nodes are whose parent. Still only O(log n) nodes touched
per operation — just not always the absolute minimum an unbalanced BST
would manage on its best day.

## Staying balanced

A plain BST's shape depends entirely on insertion order — insert
already-sorted values and it degenerates into a straight line, O(n)
deep instead of O(log n). Past about 1000 elements inserted in sorted
order, that's not just slow, it's a crash: every operation here is
recursive, and a straight-line tree blows Python's default recursion
limit.

```python
from safe_tree import EMPTY, insert, depth, size

tree = EMPTY
for i in range(100_000):
    tree = insert(tree, i)  # worst case for an unbalanced BST

size(tree)  # 100000
depth(tree)  # 17 — not 100000
```

Every `insert`/`delete` rebalances on the way back up (AVL: rotate
when a node's left and right heights differ by more than one), so
depth stays within a small constant factor of log2(size) regardless of
insertion order. `tests/test_tree.py` pins this down directly, inserting
10,000 sorted values and asserting depth stays logarithmic rather than
crashing.

## Equality

`Node` is a frozen dataclass, so `==` already works — but it compares
*shape* (and the cached `height`/`size` bookkeeping) as well as values,
because that's just what dataclass field equality does. Even with AVL
balancing, more than one valid balanced shape can exist for the same
set of values, so two trees holding the same values can still compare
unequal with `==`:

```python
by_ascending = insert(insert(insert(insert(EMPTY, 0), 1), 2), 3)
by_other_order = insert(insert(insert(insert(EMPTY, 0), 1), 3), 2)

list(in_order(by_ascending)) == list(in_order(by_other_order))  # True  — same values
by_ascending == by_other_order  # False — different (both valid) shapes
```

For "do these hold the same values," use `equal`, which compares in
sorted order and ignores shape:

```python
from safe_tree import equal

equal(by_ascending, by_other_order)  # True
```

## A slightly bigger example

The point of returning a tree from every operation is that they compose
without ceremony:

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

`tree`, `passing`, and `curved` are three different trees. Nothing about
building `curved` disturbed `passing`, and nothing about building
`passing` disturbed the original `tree` — you could keep all three
around and query any of them.

## Order statistics and range queries

Using the same `tree` of scores from above — `minimum`/`maximum` are
O(log n); `kth` uses each node's cached `size` to find the k-th
smallest value (0-indexed) in O(log n) rather than materializing a full
traversal; `range_query` prunes subtrees entirely outside the bound
instead of filtering everything, so it costs O(log n + k) for k results:

```python
from safe_tree import minimum, maximum, kth, range_query

minimum(tree)  # 45
maximum(tree)  # 91
kth(tree, 0)  # 45 — smallest
kth(tree, 3)  # 72 — 4th smallest
list(range_query(tree, 60, 90))  # [63, 72, 79, 88]
```

## Set algebra

`union`/`intersection`/`difference` build on `insert`/`member`/`filter_tree`
directly rather than a specialized merge algorithm, so they're simple
and obviously correct, at O(size(other) * log n) rather than the
O(m + n) a from-scratch tree-merge could achieve:

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

## Lenses

A `Lens[S, A]` is a frozen struct holding a pure `get` and `put` pair
focused on part `A` of a whole `S`. `then` composes two lenses so you can
reach straight through a node into a field of its own payload, without
writing the reconstruction by hand:

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
    get=lambda p: p.x,
    put=lambda p, x: Point(x, p.y),
)

node = Node(Point(1, 2), EMPTY, EMPTY)
node_x = then(value_lens(), x_lens)

node_x.get(node)  # 1
node_x.put(node, 9).value  # Point(x=9, y=2)
```

Without `then`, that `put` would mean manually destructuring `node`,
destructuring `node.value`, building a new `Point` with `x` swapped, and
rebuilding the `Node` around it. `then` turns that into one lens, used
like any other.

`value_lens`, `left_lens`, and `right_lens` are generic *functions*, not
pre-built values — call them (`value_lens()`), don't reference them
bare, so `mypy` can resolve the type parameter at each use site.

Lenses here focus on a node's own fields (`value`, `left`, `right`),
which always exist and so are always safe to get/put. A tree position
several levels down may or may not exist — that's inherently partial,
which is why there's no general `path_lens`; combine a lens with
ordinary pattern matching for that case instead.

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

Full API reference (generated from the same Google-style docstrings
shown above) via [pdoc](https://pdoc.dev):

```sh
uv run pdoc -o docs safe_tree
```

## Playground

A small local web app for watching the library's own guarantees happen
instead of just reading about them. Two independent trees (A and B),
each with their own insert/delete controls and version timeline:

- Selecting a version highlights which of its nodes are new (amber)
  versus literally the same object as in the version before it (pine
  outline) — the same thing `tests/test_sharing.py` checks with `is`.
- A timeline chip is marked **rotated** when that step's AVL
  rebalancing actually fired, detected by comparing the real result's
  shape against what a plain, non-rebalancing insert/delete would have
  produced from the same starting point.
- `min`/`max` and a `kth` lookup sit next to size/depth in the stats
  panel; a `range_query` slider highlights matching nodes directly on
  the tree.
- A **Combine** panel runs `union`/`intersection`/`difference` on
  A and B's current trees and renders the result.

```sh
uv run --group playground uvicorn playground.app:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` on the same machine, or, since it's bound
to `0.0.0.0`, `http://<your-machine's-LAN-IP>:8000` from a phone on the
same Wi-Fi network (or over something like Tailscale, from anywhere).

## Development

```sh
uv sync --all-extras --all-groups
uv run ruff check .                           # lint
uv run ruff format .                          # format
uv run mypy src tests playground conftest.py  # strict type-check
uv run pytest                                 # tests, incl. doctests, hypothesis properties, and JAX (skipped if not installed)
```

## JAX

A frozen, tagged-union tree is already close to what JAX's `pytree`
protocol expects — `jax_support.register()` makes it exact, so trees
pass transparently through `jax.tree_util.tree_map`, `jax.jit`, and
other pytree-aware transforms. `jax` is optional (`uv sync --group jax`),
not a dependency of `safe_tree` itself:

```python
import jax
from safe_tree import EMPTY, insert, in_order
from safe_tree.jax_support import register

register()  # once, after importing jax

tree = insert(insert(insert(EMPTY, 5), 2), 8)
jax.tree_util.tree_leaves(tree)  # [5, 2, 8]
list(in_order(jax.tree_util.tree_map(lambda v: v * 2, tree)))  # [4, 10, 16]
```

Structural sharing also means keeping every intermediate version of a
dataset or a split — exactly what the [playground](#playground) shows
off — is cheap in memory, a useful property for anything that wants to
checkpoint or time-travel through training state.
