from typing import Callable, FrozenSet

from pyperplan.heuristics.heuristic_base import Heuristic as PyperplanHeuristic
from pyperplan.task import Task as PyperplanTask

from hypergraph_nets.hypergraphs import HypergraphsTuple
from strips_hgn.models.strips_hgn import STRIPSHGN
from strips_hgn.planning import PlannerForEvaluation, Proposition
from strips_hgn.planning.pyperplan_api import PyperplanSupported


class STRIPSHGNHeuristic(PyperplanHeuristic, PyperplanSupported):
    """ STRIPS-HGN Heuristic Interface """

    def __init__(
        self,
        model: STRIPSHGN,
        state_to_input_hypergraphs_tuple: Callable[
            [FrozenSet[Proposition]], HypergraphsTuple
        ],
        planner: PlannerForEvaluation,
        num_steps: int = 10,
    ):
        """
        Parameters
        ----------
        model: STRIPSHGN model
        state_to_input_hypergraphs_tuple: function that maps the current state
            to a input HypergraphsTuple
        """
        # Setup prediction mode to minimise unnecessary torch evals
        model.setup_prediction_mode()
        self.model = model

        self._state_to_input_hypergraphs_tuple = (
            state_to_input_hypergraphs_tuple
        )

        self._num_steps = num_steps
        self._planner = planner

    def to_pyperplan(self, task: PyperplanTask) -> PyperplanHeuristic:
        return self

    def __call__(self, obj) -> float:
        """
        Heuristic call from the planner

        Parameters
        ----------
        obj: the object from the planner, e.g. a state, search node, etc.

        Returns
        -------
        Heuristic value
        """
        if self._planner == PlannerForEvaluation.pyperplan:
            node = obj
            # Get the hypergraphs tuple for the current state
            input_h_tuple = self._state_to_input_hypergraphs_tuple(node.state)

            # Run the STRIPS-HGN and get the predicted heuristic value
            output_h_tuple = self.model(input_h_tuple, self._num_steps)
            pred = output_h_tuple[-1].globals.item()
            return pred
        else:
            raise RuntimeError(
                f"Unsupported evaluation planner {self._planner}"
            )

    def __repr__(self):
        return "strips-hgn"

    def __str__(self):
        return self.__repr__()
