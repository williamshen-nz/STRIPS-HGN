import logging
from typing import Dict, List, Optional, Union

from pyperplan.search import SearchMetrics

import strips_hgn.planning.pyperplan_api as pyperplan_api
from strips_hgn.models.heuristic import STRIPSHGNHeuristic
from strips_hgn.planning import Heuristic, SearchAlgorithm, STRIPSProblem
from strips_hgn.utils import Number

_log = logging.getLogger(__name__)


def run_fast_downward(
    problem: STRIPSProblem,
    search_algorithm: SearchAlgorithm,
    heuristic: Heuristic,
):
    """
    Run Fast-downward for a planning task with a given search algorithm
    and heuristic.

    TODO: search metrics should ideally be passed back here or dumped in the
    C++ code
    """
    raise NotImplementedError


def evaluate_problem_with_pyperplan(
    problem: STRIPSProblem,
    search_algorithm: SearchAlgorithm,
    strips_hgn_heuristic: Optional[STRIPSHGNHeuristic],
    heuristics: List[Heuristic],
    max_search_time: Number,
) -> Dict[Union[Heuristic, STRIPSHGNHeuristic], SearchMetrics]:
    """
    Evaluate a problem with Pyperplan

    Parameters
    ----------
    problem: the STRIPS problem
    search_algorithm: search algorithm to use
    strips_hgn_heuristic: the Pyperplan heuristic for a STRIPS-HGN
    heuristics: all the heuristics to compare against
    max_search_time: maximum search time for the problem

    Returns
    -------
    A dict mapping a heuristic name to the resulting metrics from the search
    """
    if not (strips_hgn_heuristic or heuristics):
        raise ValueError(
            "Either `strips_hgn_heuristic` or `heuristics` must be specified"
        )

    # Get the Pyperplan task
    _, task = pyperplan_api.get_domain_and_task(
        problem.domain_pddl, problem.problem_pddl
    )

    pyperplan_search_algorithm = search_algorithm.to_pyperplan(task)

    # Resulting metrics dict
    heuristic_to_metrics = dict()

    # Run the search for each non-learned heuristic + STRIPS-HGN heuristic
    for heuristic in (*heuristics, strips_hgn_heuristic):
        _log.info(f"Running {search_algorithm} + {heuristic} with pyperplan")
        _, metrics = pyperplan_api.find_solution(
            task=task,
            heuristic=heuristic.to_pyperplan(task),
            search_algo=pyperplan_search_algorithm,
            max_search_time=max_search_time,
        )
        heuristic_to_metrics[heuristic] = metrics

    # Sanity check
    assert all(heuristic in heuristic_to_metrics for heuristic in heuristics)
    if strips_hgn_heuristic:
        assert strips_hgn_heuristic in heuristic_to_metrics

    return heuristic_to_metrics
