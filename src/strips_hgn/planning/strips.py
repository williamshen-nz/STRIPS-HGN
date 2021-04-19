from abc import ABC, abstractmethod
from typing import FrozenSet, List, NamedTuple, TypeVar

from pyperplan.pddl.pddl import Domain as PyperplanDomain
from pyperplan.task import Task as PyperplanTask

from strips_hgn.planning.pyperplan_api import get_domain_and_task
from strips_hgn.utils import Number

# Proposition - i.e. fact, atom
Proposition = TypeVar("Proposition", bound=str)


# STRIPSAction which essentially represents a hyperedge
class STRIPSAction(NamedTuple):
    name: str
    cost: Number
    preconditions: FrozenSet[Proposition]
    add_effects: FrozenSet[Proposition]
    del_effects: FrozenSet[Proposition]

    def apply(self, state: FrozenSet[Proposition]) -> FrozenSet[Proposition]:
        """
        Apply an action in a given state to get the new state.

        Parameters
        ----------
        state: FrozenSet[Proposition], current state

        Returns
        -------
        FrozenSet[Proposition], new state after applying this action
        """
        # Check we can actually apply this action
        if not self.preconditions.issubset(state):
            raise RuntimeError(f"Cannot apply {self.name} in state {state}")

        # Compute new state: (s \ Del(o)) U Add(o)
        return (state.difference(self.del_effects)).union(self.add_effects)


class STRIPSProblem(ABC):
    """ Abstract STRIPS Problem """

    def __init__(self, domain_pddl: str, problem_pddl: str):
        self.domain_pddl = domain_pddl
        self.problem_pddl = problem_pddl

    @property
    @abstractmethod
    def domain_name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def initial_state(self) -> FrozenSet[Proposition]:
        raise NotImplementedError

    @property
    @abstractmethod
    def goals(self) -> FrozenSet[Proposition]:
        raise NotImplementedError

    def is_goal_state(self, state: FrozenSet[Proposition]) -> bool:
        """ Whether the given state is a goal state """
        return self.goals.issubset(state)

    @property
    @abstractmethod
    def propositions(self) -> List[Proposition]:
        raise NotImplementedError

    @property
    def number_of_propositions(self) -> int:
        return len(self.propositions)

    @property
    @abstractmethod
    def actions(self) -> List[STRIPSAction]:
        raise NotImplementedError

    @property
    def number_of_actions(self) -> int:
        return len(self.actions)


class _PyperplanSTRIPSProblem(STRIPSProblem):
    """
    STRIPS Problem implemented with Pyperplan
    """

    def __init__(self, domain_pddl: str, problem_pddl: str):
        super().__init__(domain_pddl, problem_pddl)
        # Setup pyperplan domain and task
        domain, task = get_domain_and_task(domain_pddl, problem_pddl)
        self._pyperplan_domain: PyperplanDomain = domain
        self._pyperplan_task: PyperplanTask = task

        # Mapping of proposition to a unique ID
        self._proposition_to_idx = {}
        self._propositions = []

        for idx, proposition in enumerate(self._pyperplan_task.facts):
            self._proposition_to_idx[proposition] = idx
            self._propositions.append(proposition)

        # Pyperplan only supports unit cost actions
        self._actions = [
            STRIPSAction(
                op.name, 1.0, op.preconditions, op.add_effects, op.del_effects
            )
            for op in self._pyperplan_task.operators
        ]

    @property
    def domain_name(self) -> str:
        return self._pyperplan_domain.name

    @property
    def name(self) -> str:
        return self._pyperplan_task.name

    @property
    def initial_state(self) -> FrozenSet[Proposition]:
        return self._pyperplan_task.initial_state

    @property
    def goals(self) -> FrozenSet[Proposition]:
        return self._pyperplan_task.goals

    @property
    def propositions(self) -> List[Proposition]:
        return self._propositions

    @property
    def actions(self) -> List[STRIPSAction]:
        return self._actions


def get_strips_problem(
    domain_pddl: str, problem_pddl: str, use_pyperplan=True
):
    """
    A factory-ish pattern to abstract the underlying implementation using
    Pyperplan away
    """
    if not use_pyperplan:
        raise NotImplementedError(
            "Only Pyperplan is supported for generating STRIPSProblem objects "
            "at the moment!"
        )

    return _PyperplanSTRIPSProblem(
        domain_pddl=domain_pddl, problem_pddl=problem_pddl
    )
