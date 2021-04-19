import pytest

from strips_hgn.planning.strips import STRIPSAction


_TEST_ACTION = STRIPSAction(
    name="test",
    cost=1.0,
    preconditions=frozenset({"a", "b", "c"}),
    add_effects=frozenset({"d", "e"}),
    del_effects=frozenset({"b", "c", "f"}),
)


@pytest.mark.parametrize(
    "current_state, expected_new_state",
    [
        (frozenset({"a", "b", "c"}), frozenset({"a", "d", "e"})),
        (
            frozenset({"a", "b", "c", "f", "g"}),
            frozenset({"a", "d", "e", "g"}),
        ),
        (
            frozenset({"a", "b", "c", "d", "e", "f", "g", "h"}),
            frozenset({"a", "d", "e", "g", "h"}),
        ),
    ],
)
def test_strips_action(current_state, expected_new_state):
    """
    Check STRIPSAction correctly applies an action in a given state
    """
    assert _TEST_ACTION.apply(current_state) == expected_new_state


@pytest.mark.parametrize(
    "current_state",
    [frozenset({"a", "b", "d", "e"}), frozenset(), frozenset({"a", "b"})],
)
def test_strips_action_raises_error(current_state):
    """
    Check STRIPSAction raises error if we try to apply an action in a
    state where the preconditions are not satisfied
    """
    with pytest.raises(RuntimeError):
        _TEST_ACTION.apply(current_state)
