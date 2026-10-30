"""A local playground for exploring safe_tree interactively.

Run with:

    uv run --group playground uvicorn playground.app:app --host 0.0.0.0 --port 8000

Every insert/delete appends a new version to an in-memory history
instead of mutating the previous one, so the whole point of the
library — old versions staying valid, new versions sharing structure
with them — is visible in the UI rather than just asserted in tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from safe_tree import EMPTY, Empty, Node, Tree, delete, depth, in_order, insert, size

STATIC_DIR = Path(__file__).parent / "static"


@dataclass(frozen=True, slots=True)
class Version:
    """One snapshot in the playground's insert/delete history."""

    label: str
    tree: Tree[int]


class ValueRequest(BaseModel):
    """Request body carrying the value to insert or delete."""

    value: int


class VersionView(BaseModel):
    """JSON shape of one history entry, as sent to the frontend."""

    index: int
    label: str
    tree: dict[str, Any] | None
    node_ids: list[str]
    size: int
    depth: int
    values: list[int]


class HistoryView(BaseModel):
    """JSON shape of the full history, as sent to the frontend."""

    versions: list[VersionView]


app = FastAPI(title="safe_tree playground")

history: list[Version] = [Version("EMPTY", EMPTY)]


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


def _collect_ids(tree: Tree[int]) -> list[str]:
    """List the identity of every node in ``tree``."""
    match tree:
        case Empty():
            return []
        case Node(_, left, right):
            return [str(id(tree)), *_collect_ids(left), *_collect_ids(right)]


def _version_view(index: int) -> VersionView:
    """Build the frontend-facing view of ``history[index]``."""
    version = history[index]
    return VersionView(
        index=index,
        label=version.label,
        tree=_tree_json(version.tree),
        node_ids=_collect_ids(version.tree),
        size=size(version.tree),
        depth=depth(version.tree),
        values=list(in_order(version.tree)),
    )


def _history_view() -> HistoryView:
    """Build the frontend-facing view of the entire history."""
    return HistoryView(versions=[_version_view(i) for i in range(len(history))])


@app.get("/api/state")
def get_state() -> HistoryView:
    """Return the full version history."""
    return _history_view()


@app.post("/api/insert")
def post_insert(body: ValueRequest) -> HistoryView:
    """Append a new version with ``body.value`` inserted."""
    history.append(Version(f"insert({body.value})", insert(history[-1].tree, body.value)))
    return _history_view()


@app.post("/api/delete")
def post_delete(body: ValueRequest) -> HistoryView:
    """Append a new version with ``body.value`` deleted."""
    history.append(Version(f"delete({body.value})", delete(history[-1].tree, body.value)))
    return _history_view()


@app.post("/api/reset")
def post_reset() -> HistoryView:
    """Reset the history back to a single `EMPTY` version."""
    history[:] = [Version("EMPTY", EMPTY)]
    return _history_view()


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
