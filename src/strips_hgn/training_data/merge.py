import logging
from collections import defaultdict
from typing import Dict, List

from strips_hgn.planning import STRIPSProblem
from strips_hgn.training_data import StateValuePair, TrainingPair
from strips_hgn.utils.metrics import CountMetric, metrics_logger

_log = logging.getLogger(__name__)


def merge_state_value_pairs_by_domain(
    problem_to_state_value_pairs: Dict[STRIPSProblem, List[StateValuePair]],
    remove_duplicates: bool = False,
) -> Dict[str, List[TrainingPair]]:
    """
    Generates a mapping of domain to corresponding TrainingPairs.
    The state-value pairs are merged by domain, and corresponding TrainingPair
    objects are created.

    The TrainingPair objects contain the problem, which we use to generate the
    hypergraph later on.

    Parameters
    ----------
    problem_to_state_value_pairs: mapping of STRIPSProblem to a list of
        state-value pairs
    remove_duplicates: whether to remove duplicate TrainingPairs, not
        implemented at the moment

    Returns
    -------
    Mapping of domain name to List[TrainingPair]
    """
    # Domain to training pairs. We determine a unique domain by its name
    domain_to_training_pairs = defaultdict(list)

    for problem, state_value_pairs in problem_to_state_value_pairs.items():
        # Create TrainingPair objects which hold the problem context
        training_pairs = [
            TrainingPair(problem, state_value_pair)
            for state_value_pair in state_value_pairs
        ]
        domain_to_training_pairs[problem.domain_name].extend(training_pairs)

    if remove_duplicates:
        # TODO: figure out best way to implement this
        # Options: (option 2 is strongly preferred)
        #  1. Remove duplicates based on state and value only
        #  2. Remove duplicates based on hypergraph structure, state and value
        raise NotImplementedError

    # Metrics
    total_num_pairs = 0
    for domain, training_pairs in domain_to_training_pairs.items():
        metrics_logger.add_metric(
            CountMetric(
                "NumberofMergedTrainingPairs",
                len(training_pairs),
                context={"domain": domain},
            )
        )
        _log.debug(
            f"Merged {len(training_pairs)} training pairs for '{domain}'"
        )
        total_num_pairs += len(training_pairs)

    _log.info(f"Merged {total_num_pairs} training pairs in total")
    metrics_logger.add_metric(
        CountMetric("TotalNumberOfMergedTrainingPairs", total_num_pairs)
    )

    return domain_to_training_pairs
