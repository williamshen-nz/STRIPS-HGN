import pytest

from strips_hgn.features.node_features import PropositionInStateAndGoal

_CURRENT_STATE = frozenset({"a", "b", "c"})
_GOAL_STATE = frozenset({"b", "c", "d"})


@pytest.mark.parametrize(
    "proposition, current_state, goal_state, expected_features",
    [
        # Proposition in current state only
        ("a", _CURRENT_STATE, _GOAL_STATE, [1, 0]),
        # Proposition in goal state only
        ("d", _CURRENT_STATE, _GOAL_STATE, [0, 1]),
        # Proposition in current and goal state
        ("c", _CURRENT_STATE, _GOAL_STATE, [1, 1]),
    ],
)
def test_proposition_in_state_or_goal(
    proposition, current_state, goal_state, expected_features
):
    feature_mapper = PropositionInStateAndGoal(current_state, goal_state)
    assert feature_mapper(proposition) == expected_features
