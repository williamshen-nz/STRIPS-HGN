import logging
from typing import FrozenSet

from strips_hgn.planning import Proposition, STRIPSProblem
from strips_hgn.utils import Number

"""
The training_data submodule is divided into five key components

1. Generating training data (state-value pairs), generate.py
2. Merging training data from the different problems, merge.py
3. Post-processing training data, process.py
4. Saving training data, save.py

workflow.py runs steps 1 to 4

The code for loading existing training data will be implemented in the future.
"""

_log = logging.getLogger(__name__)


class StateValuePair(object):
    """
    A pair which consists of a state and heuristic value.
    This can't be a NamedTuple otherwise the default json encoder complains.
    """

    def __init__(self, state: FrozenSet[Proposition], value: Number):
        self.state: FrozenSet[Proposition] = state
        self.value: Number = value

    def __repr__(self):
        return f"StateValuePair(state={self.state}, value={self.value})"

    def to_json_dict(self) -> dict:
        # For saving training data
        json_dict = {
            "state": [prop for prop in self.state],
            "value": self.value,
        }
        return json_dict


class TrainingPair(StateValuePair):
    """
    A training pair which consists of a problem, a state, and the
    target heuristic value for the state.

    StateValuePair and TrainingPair are different classes for easier
    extensibility in the future when we move to experimenting with other
    target heuristics, not just the perfect heuristic h*.
    """

    def __init__(
        self, problem: STRIPSProblem, state_value_pair: StateValuePair
    ):
        super().__init__(state_value_pair.state, state_value_pair.value)
        self.problem = problem

    def __repr__(self):
        return (
            f"TrainingPair(problem={self.problem.name}, "
            f"state={self.state}, value={self.value})"
        )

    def to_json_dict(self) -> dict:
        # For saving training data
        json_dict = super().to_json_dict()
        json_dict["problem"] = {
            "name": self.problem.name,
            "domain_name": self.problem.domain_name,
            "domain_pddl": self.problem.domain_pddl,
            "problem_pddl": self.problem.problem_pddl,
        }
        return json_dict
