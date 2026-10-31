"""A local playground for exploring safe_tree interactively.

Run with:

    uv run --group playground uvicorn playground.app:app --host 0.0.0.0 --port 8000

Every insert/delete appends a new version to an in-memory history
instead of mutating the previous one, so the whole point of the
library — old versions staying valid, new versions sharing structure
with them — is visible in the UI rather than just asserted in tests.

Two independent trees (`a`/`b`) let the UI show off `union`/
`intersection`/`difference` between them, on top of the single-tree
story (structural sharing, AVL rebalancing, order statistics).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from safe_tree import (
    EMPTY,
    Empty,
    Node,
    Tree,
    delete,
    depth,
    difference,
    in_order,
    insert,
    intersection,
    size,
    union,
)

STATIC_DIR = Path(__file__).parent / "static"

TreeId = Literal["a", "b"]
CombineOp = Literal["union", "intersection", "difference"]

# A plain, non-rebalancing BST used only as a comparison baseline, to
# detect whether safe_tree's AVL rebalancing actually fired for a given
# operation. Not part of the library; this is diagnostic plumbing for
# the playground's UI. Detection compares *shapes* (this naive tree
# against the real tree with its bookkeeping stripped), not node
# counts: a single rotation can touch exactly as many final nodes as
# plain path-copying would, just arranged differently, so counting
# allocations doesn't reliably tell rotated apart from not-rotated.
type NaiveTree = tuple[int, "NaiveTree | None", "NaiveTree | None"] | None


def _naive_insert(tree: NaiveTree, value: int) -> NaiveTree:
    """Insert into a naive BST, no rebalancing."""
    if tree is None:
        return (value, None, None)
    v, left, right = tree
    if value < v:
        return (v, _naive_insert(left, value), right)
    if v < value:
        return (v, left, _naive_insert(right, value))
    return (value, left, right)


def _naive_delete_min(tree: tuple[int, NaiveTree, NaiveTree]) -> tuple[int, NaiveTree]:
    """Remove and return the smallest value from a non-empty naive BST."""
    v, left, right = tree
    if left is None:
        return v, right
    min_value, new_left = _naive_delete_min(left)
    return min_value, (v, new_left, right)


def _naive_delete(tree: NaiveTree, value: int) -> NaiveTree:
    """Delete from a naive BST, no rebalancing."""
    if tree is None:
        return None
    v, left, right = tree
    if value < v:
        return (v, _naive_delete(left, value), right)
    if v < value:
        return (v, left, _naive_delete(right, value))
    if left is None:
        return right
    if right is None:
        return left
    successor, new_right = _naive_delete_min(right)
    return (successor, left, new_right)


def _shape(tree: Tree[int]) -> NaiveTree:
    """Strip a real tree down to a plain (value, left, right) shape."""
    match tree:
        case Empty():
            return None
        case Node(value, left, right):
            return (value, _shape(left), _shape(right))


def _tree_json(tree: Tree[int]) -> dict[str, Any] | None:
    """Serialize a tree to nested JSON, keyed by Python object identity.

    The identity (``id(tree)``) is what lets the frontend tell, across
    two versions, whether a node is the exact same object (shared) or a
    freshly allocated one (new) — the same guarantee ``tests/test_sharing.py``
    checks with ``is``.
    """
    match tree:
        case Empty():
            return None
        case Node(value, left, right):
            return {
                "id": str(id(tree)),
                "value": value,
                "left": _tree_json(left),
                "right": _tree_json(right),
            }


def _collect_ids(tree: Tree[int]) -> set[str]:
    """Collect the identity of every node in ``tree``."""
    match tree:
        case Empty():
            return set()
        case Node(_, left, right):
            return {str(id(tree)), *_collect_ids(left), *_collect_ids(right)}


@dataclass(frozen=True, slots=True)
class Version:
    """One snapshot in a tree's insert/delete history."""

    label: str
    tree: Tree[int]
    rotated: bool


class ValueRequest(BaseModel):
    """Request body carrying the value to insert or delete."""

    value: int


class CombineRequest(BaseModel):
    """Request body naming which set operation to combine A and B with."""

    op: CombineOp


class TreeView(BaseModel):
    """JSON shape of a single tree snapshot, as sent to the frontend."""

    label: str
    tree: dict[str, Any] | None
    node_ids: list[str]
    size: int
    depth: int
    values: list[int]
    minimum: int | None
    maximum: int | None


class VersionView(TreeView):
    """A `TreeView` that's also a numbered step in a tree's history."""

    index: int
    rotated: bool


class HistoryView(BaseModel):
    """JSON shape of a tree's entire version history."""

    versions: list[VersionView]


app = FastAPI(title="safe_tree playground")

histories: dict[TreeId, list[Version]] = {
    "a": [Version("EMPTY", EMPTY, rotated=False)],
    "b": [Version("EMPTY", EMPTY, rotated=False)],
}


def _tree_view(label: str, tree: Tree[int]) -> TreeView:
    """Build the frontend-facing view of a single tree."""
    values = list(in_order(tree))
    return TreeView(
        label=label,
        tree=_tree_json(tree),
        node_ids=sorted(_collect_ids(tree)),
        size=size(tree),
        depth=depth(tree),
        values=values,
        minimum=values[0] if values else None,
        maximum=values[-1] if values else None,
    )


def _version_view(tree_id: TreeId, index: int) -> VersionView:
    """Build the frontend-facing view of one step in ``tree_id``'s history."""
    version = histories[tree_id][index]
    base = _tree_view(version.label, version.tree)
    return VersionView(index=index, rotated=version.rotated, **base.model_dump())


def _history_view(tree_id: TreeId) -> HistoryView:
    """Build the frontend-facing view of ``tree_id``'s entire history."""
    return HistoryView(versions=[_version_view(tree_id, i) for i in range(len(histories[tree_id]))])


@app.get("/api/state/{tree_id}")
def get_state(tree_id: TreeId) -> HistoryView:
    """Return ``tree_id``'s full version history."""
    return _history_view(tree_id)


@app.post("/api/{tree_id}/insert")
def post_insert(tree_id: TreeId, body: ValueRequest) -> HistoryView:
    """Append a new version with ``body.value`` inserted into ``tree_id``.

    Flags ``rotated`` by comparing the real result's shape against what
    a plain, non-rebalancing insert into the *same starting shape*
    would have produced — comparing against a baseline that's reset on
    every call, since two shapes that already differ (from an earlier
    rotation) would otherwise keep "differing" on every later step
    regardless of whether that step itself needed to rebalance.
    """
    current_tree = histories[tree_id][-1].tree
    new_tree = insert(current_tree, body.value)
    rotated = _shape(new_tree) != _naive_insert(_shape(current_tree), body.value)
    histories[tree_id].append(Version(f"insert({body.value})", new_tree, rotated=rotated))
    return _history_view(tree_id)


@app.post("/api/{tree_id}/delete")
def post_delete(tree_id: TreeId, body: ValueRequest) -> HistoryView:
    """Append a new version with ``body.value`` deleted from ``tree_id`` (see `post_insert`)."""
    current_tree = histories[tree_id][-1].tree
    new_tree = delete(current_tree, body.value)
    rotated = _shape(new_tree) != _naive_delete(_shape(current_tree), body.value)
    histories[tree_id].append(Version(f"delete({body.value})", new_tree, rotated=rotated))
    return _history_view(tree_id)


@app.post("/api/{tree_id}/reset")
def post_reset(tree_id: TreeId) -> HistoryView:
    """Reset ``tree_id``'s history back to a single `EMPTY` version."""
    histories[tree_id] = [Version("EMPTY", EMPTY, rotated=False)]
    return _history_view(tree_id)


@app.post("/api/combine")
def post_combine(body: CombineRequest) -> TreeView:
    """Combine the latest versions of trees A and B with ``body.op``."""
    tree_a = histories["a"][-1].tree
    tree_b = histories["b"][-1].tree
    op_fn = {"union": union, "intersection": intersection, "difference": difference}[body.op]
    op_label = {"union": "A | B", "intersection": "A & B", "difference": "A - B"}[body.op]
    return _tree_view(op_label, op_fn(tree_a, tree_b))


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
