import logging
from typing import Dict, List, Optional

from strips_hgn.config import TRAINING_DATA_TIMER_LOG_LEVEL
from strips_hgn.planning import STRIPSAction, STRIPSProblem
from strips_hgn.planning.fast_downward_api import get_optimal_actions_using_fd
from strips_hgn.training_data import StateValuePair
from strips_hgn.utils.metrics import CountMetric, metrics_logger
from strips_hgn.utils.timer import TimedOperation, timed

_log = logging.getLogger(__name__)


def _generate_optimal_state_value_pairs_for_problem(
    problem: STRIPSProblem
) -> List[StateValuePair]:
    """
    Generates the optimal state-value pairs for a planning problem.

    Parameters
    ----------
    problem: STRIPSProblem, the problem we are generating state-value pairs for

    Returns
    -------
    List[StateValuePair] with the trajectories of the states and optimal
    heuristic values
    """
    # Start a timer
    metric_context = {"domain": problem.domain_name, "problem": problem.name}
    timer = TimedOperation(
        "GenerateOptimalStateValuePairsTime",
        context=metric_context,
        log_level=TRAINING_DATA_TIMER_LOG_LEVEL,
    ).start()

    # Run Fast-Downward to get the optimal plan
    optimal_plan: Optional[List[str]] = get_optimal_actions_using_fd(problem)

    # Check some edge cases
    if len(optimal_plan) == 0:
        _log.warning(f"Initial state for {problem} is already a goal state!")
        return []
    elif optimal_plan is None:
        _log.error(f"Unable to find optimal solution for {problem}")
        return []

    name_to_action: Dict[str, STRIPSAction] = {
        action.name: action for action in problem.actions
    }
    # Form state-value pairs for trajectory which is at first the initial state
    current_state = problem.initial_state
    trajectory: List[StateValuePair] = [
        StateValuePair(current_state, len(optimal_plan))
    ]

    for idx, action_name in enumerate(optimal_plan):
        # Apply action in the current state
        action = name_to_action[action_name]
        current_state = action.apply(current_state)

        # Create new state-value pair
        remaining_plan_length = len(optimal_plan) - (idx + 1)
        trajectory.append(StateValuePair(current_state, remaining_plan_length))

    # Check current state is a goal state and the number of pairs
    assert problem.is_goal_state(current_state)
    assert len(trajectory) == len(optimal_plan) + 1

    # Stop timer and add metric for number of state-value pairs
    timer.stop()
    metrics_logger.add_metric(
        CountMetric(
            "NumberOfOptimalStateValuePairs",
            len(trajectory),
            context=metric_context,
        )
    )
    return trajectory


@timed(
    "GenerateAllOptimalStateValuePairsTime",
    log_level=TRAINING_DATA_TIMER_LOG_LEVEL,
)
def generate_optimal_state_value_pairs(
    problems: List[STRIPSProblem]
) -> Dict[STRIPSProblem, List[StateValuePair]]:
    """
    Generate the state-value pairs from the optimal plans of each task by using
    Fast Downward.

    Parameters
    ----------
    problems: List[STRIPSProblem], the problems to generate optimal
    state-value pairs for

    Returns
    -------
    Dict[STRIPSProblem, List[StateValuePair]], a mapping of each problem to the
    state-value pairs encountered on the optimal plan.
    """
    training_data: Dict[STRIPSProblem, List[StateValuePair]] = {}
    total_num_state_value_pairs = 0

    for problem in problems:
        if problem in training_data:
            raise RuntimeError(
                f"Already generated optimal state-value pairs for"
                f" {problem.name}"
            )

        # Generate state-value pairs for each problem
        state_value_pairs = _generate_optimal_state_value_pairs_for_problem(
            problem
        )

        _log.debug(
            f"Generated {len(state_value_pairs)} state-value pairs for "
            f"{problem.name}"
        )
        training_data[problem] = state_value_pairs
        total_num_state_value_pairs += len(state_value_pairs)

    _log.info(
        f"Generated {total_num_state_value_pairs} state-value pairs in "
        f"total for {len(problems)} tasks"
    )
    metrics_logger.add_metric(
        CountMetric(
            "TotalNumberOfStateValuePairs", total_num_state_value_pairs
        )
    )
    return training_data
