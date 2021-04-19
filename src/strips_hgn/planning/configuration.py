from abc import abstractmethod
from enum import Enum
from typing import Set

from pyperplan.heuristics.heuristic_base import Heuristic as PyperplanHeuristic
from pyperplan.task import Task as PyperplanTask

from strips_hgn.planning.pyperplan_api import (
    HEURISTIC_STR_TO_CLASS,
    SEARCH_ALGO_STR_TO_FUNC,
    PyperplanSupportedEnum,
)


class _BasePlanningEnum(Enum):
    @classmethod
    def member_names(cls) -> Set[str]:
        return {member.name for member in cls}

    @classmethod
    @abstractmethod
    def from_str(cls, the_str: str):
        raise NotImplementedError

    @classmethod
    def _from_str(cls, the_str: str, error_str="Unknown member {}"):
        for member in cls:
            if member.name == the_str:
                return member

        raise ValueError(error_str.format(the_str))

    def __str__(self):
        return self.value


class SearchAlgorithm(_BasePlanningEnum, PyperplanSupportedEnum):
    """ Search Algorithm """

    a_star = "a-star"

    def to_pyperplan(self, _: PyperplanTask):
        if self.value not in SEARCH_ALGO_STR_TO_FUNC:
            raise ValueError(
                f"Unsupported Pyperplan search algorithm {self.value}"
            )

        return SEARCH_ALGO_STR_TO_FUNC[self.value]

    @classmethod
    def from_str(cls, search_algorithm_str: str) -> "SearchAlgorithm":
        return cls._from_str(
            search_algorithm_str, "Unknown search algorithm {}"
        )


class Heuristic(_BasePlanningEnum, PyperplanSupportedEnum):
    """ Planning Heuristics """

    h_max = "h-max"
    h_add = "h-add"
    lm_cut = "lm-cut"
    h_ff = "h-ff"
    blind = "blind"

    def to_pyperplan(self, task: PyperplanTask) -> PyperplanHeuristic:
        if self.value not in HEURISTIC_STR_TO_CLASS:
            raise ValueError(f"Unsupported Pyperplan heuristic {self.value}")
        return HEURISTIC_STR_TO_CLASS[self.value](task)

    @classmethod
    def from_str(cls, heuristic_str: str) -> "Heuristic":
        return cls._from_str(heuristic_str, "Unknown heuristic {}")


class PlannerForEvaluation(_BasePlanningEnum):
    """ Supported Planners for Evaluation """

    pyperplan = "pyperplan"
    # TODO: uncomment after we support Fast Downward
    # fast_downward = "fd"

    @classmethod
    def from_str(cls, planner_str: str) -> "PlannerForEvaluation":
        return cls._from_str(planner_str, "Unknown planner {}")
