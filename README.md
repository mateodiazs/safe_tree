# safe_tree

A purely functional, immutable tree for Python. Every operation returns a
new tree instead of mutating the one you have, so old versions stay valid
forever — and because updates share structure instead of copying it, that
safety doesn't cost you a full copy on every change.

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
    ├── value : A
    ├── left  : Tree[A]
    └── right : Tree[A]
```

`EMPTY` is never mutated or recreated, so emptiness is always the same
object — that's what makes `delete(tree, missing_value) is tree` true
rather than merely equal.

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
| `insert(tree, value)` | New tree with `value` added. |
| `delete(tree, value)` | New tree with `value` removed. |
| `member(tree, value)` | Membership test. |
| `map_tree(tree, f)` | Structure-preserving map over every value. |
| `fold(tree, f, init)` | In-order reduction to a single value. |
| `filter_tree(tree, pred)` | Rebuild keeping only matching values. |
| `in_order` / `pre_order` / `post_order` | Lazy traversal iterators. |
| `size(tree)` / `depth(tree)` | Structural measures. |
| `Lens`, `then`, `value_lens`, `left_lens`, `right_lens` | Composable node accessors. |

Full API reference (generated from the same Google-style docstrings
shown above) via [pdoc](https://pdoc.dev):

```sh
uv run pdoc -o docs safe_tree
```

## Development

```sh
uv sync --all-extras
uv run ruff check .      # lint
uv run ruff format .     # format
uv run mypy src tests    # strict type-check
uv run pytest            # tests, incl. doctests and hypothesis properties
```

## Looking toward ML

Nothing here is ML-specific yet, but the shape is deliberately kept
compatible with it: a frozen, tagged-union tree is close to what JAX's
`pytree` / `optree` protocols expect, and structural sharing makes
keeping every intermediate version of a dataset or a split cheap in
memory. Not in scope today — noted so early API choices don't foreclose
it later.
