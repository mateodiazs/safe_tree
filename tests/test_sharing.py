from safe_tree import EMPTY, Node, Tree, delete, insert


def test_insert_shares_the_untouched_sibling_subtree() -> None:
    tree = insert(insert(insert(EMPTY, 5), 2), 8)
    updated = insert(tree, 7)

    assert isinstance(tree, Node)
    assert isinstance(updated, Node)
    assert tree.left is updated.left


def test_reinserting_the_same_value_is_a_no_op() -> None:
    value = 5
    once = insert(EMPTY, value)
    twice = insert(once, value)

    assert twice is once


def test_deleting_an_absent_value_returns_the_same_tree() -> None:
    tree: Tree[int] = EMPTY
    for v in (1, 2, 3, 4, 5):
        tree = insert(tree, v)

    assert delete(tree, 99) is tree


def test_delete_shares_the_untouched_sibling_subtree() -> None:
    tree = insert(insert(insert(EMPTY, 5), 2), 8)
    updated = delete(tree, 8)

    assert isinstance(tree, Node)
    assert isinstance(updated, Node)
    assert tree.left is updated.left
