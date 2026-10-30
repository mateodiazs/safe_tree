"""safe_tree: a purely functional, immutable tree."""

from safe_tree.lens import Lens, left_lens, right_lens, then, value_lens
from safe_tree.tree import (
    EMPTY,
    Empty,
    Node,
    Tree,
    delete,
    depth,
    filter_tree,
    fold,
    in_order,
    insert,
    map_tree,
    member,
    post_order,
    pre_order,
    size,
)

__all__ = [
    "EMPTY",
    "Empty",
    "Lens",
    "Node",
    "Tree",
    "delete",
    "depth",
    "filter_tree",
    "fold",
    "in_order",
    "insert",
    "left_lens",
    "map_tree",
    "member",
    "post_order",
    "pre_order",
    "right_lens",
    "size",
    "then",
    "value_lens",
]
