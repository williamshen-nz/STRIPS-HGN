import json
import logging
import os
from typing import Dict, List

from hypergraph_nets.hypergraphs import HypergraphsTuple
from strips_hgn.hypergraph.delete_relaxation import (
    DeleteRelaxationHypergraphView,
)
from strips_hgn.models.heuristic import STRIPSHGNHeuristic
from strips_hgn.models.strips_hgn import STRIPSHGN
from strips_hgn.planning import (
    Heuristic,
    PlannerForEvaluation,
    SearchAlgorithm,
    STRIPSProblem,
)
from strips_hgn.planning.evaluation import evaluate_problem_with_pyperplan
from strips_hgn.utils import Number
from strips_hgn.utils.json_encoders import MetricsEncoder
from strips_hgn.utils.timer import timed
from strips_hgn.workflows.base_workflow import BaseFeatureMappingWorkflow

_log = logging.getLogger(__name__)


class EvaluateSTRIPSHGNWorkflow(BaseFeatureMappingWorkflow):
    """
    Workflow for evaluating a STRIPS-HGN against other heuristics for a
    set of problems
    """

    _TEST_RESULTS_NAME = "test_results.json"

    def __init__(
        self,
        model: STRIPSHGN,
        heuristics: List[Heuristic],
        search_algorithm: SearchAlgorithm,
        planner: PlannerForEvaluation,
        max_search_time: Number,
        experiments_dir: str,
    ):
        """
        Parameters
        ----------
        model: the STRIPS-HGN
        heuristics: list of Heuristics to compare the STRIPS-HGN against
        search_algorithm: search algorithm to run for the analysis
        planner: planner to use for evaluation
        max_search_time: maximum search time for each problem, in seconds
        experiments_dir: directory to add experiment results
        """
        # We can get the feature mappers from the model hyperparameters
        hparams = model.hparams

        super().__init__(
            global_feature_mapper_cls=hparams.global_feature_mapper_cls,
            node_feature_mapper_cls=hparams.node_feature_mapper_cls,
            hyperedge_feature_mapper_cls=hparams.hyperedge_feature_mapper_cls,
            max_receivers=model.hparams.receiver_k,
            max_senders=model.hparams.sender_k,
        )
        # Setup model for prediction
        model.eval()
        model.setup_prediction_mode()
        self._model = model

        # Store other fields
        self._heuristics = heuristics
        self._search_algorithm = search_algorithm
        self._planner = planner
        self._max_search_time = max_search_time
        self._experiments_dir = experiments_dir

        # Metrics
        self._evaluation_metrics: Dict[str, Dict] = {
            "configuration": {
                "heuristics": [*self._heuristics, "strips-hgn"],
                "search_algorithm": self._search_algorithm,
                "max_search_time": self._max_search_time,
                "planner": self._planner,
            },
            "results": dict(),
        }

    def _write_metrics_to_disk(self):
        """ Write the evaluation metrics to the experiments directory """
        # Write results to disk
        test_results_fname = os.path.join(
            self._experiments_dir, EvaluateSTRIPSHGNWorkflow._TEST_RESULTS_NAME
        )
        json.dump(
            self._evaluation_metrics,
            open(test_results_fname, "w"),
            indent=2,
            cls=MetricsEncoder,
        )
        _log.info(f"Wrote evaluation workflow metrics to {test_results_fname}")

    def _get_strips_hgn_heuristic(
        self, problem: STRIPSProblem
    ) -> STRIPSHGNHeuristic:
        """
        Form hypergraph and mapper from state to HypergraphsTuple and
        create the learned heuristic object
        """
        hypergraph = DeleteRelaxationHypergraphView(problem)

        def state_to_input_h_tup(state) -> HypergraphsTuple:
            return self._get_input_hypergraphs_tuple(state, hypergraph)

        return STRIPSHGNHeuristic(
            self._model, state_to_input_h_tup, self._planner
        )

    @timed("EvaluateSTRIPSHGNWorkflow.TotalTime")
    def run(self, problems: List[STRIPSProblem]):
        _log.info(f"Running evaluation workflow for {len(problems)} problems")
        _log.info(
            f"Max search time for all problems = {self._max_search_time}s"
        )

        # Run search for all the problems
        for idx, problem in enumerate(problems):
            _log.info(
                f"Evaluating {problem.name} for {problem.domain_name} "
                f"({idx + 1}/{len(problems)})"
            )

            # Get the STRIPS-HGN learned heuristic
            strips_hgn_heuristic = self._get_strips_hgn_heuristic(problem)

            if self._planner == PlannerForEvaluation.pyperplan:
                # Run pyperplan
                heuristic_to_metrics = evaluate_problem_with_pyperplan(
                    problem=problem,
                    search_algorithm=self._search_algorithm,
                    strips_hgn_heuristic=strips_hgn_heuristic,
                    heuristics=self._heuristics,
                    max_search_time=self._max_search_time,
                )

                # Store metrics for this problem
                self._evaluation_metrics["results"][problem.name] = {
                    str(heuristic): metrics._asdict()
                    for heuristic, metrics in heuristic_to_metrics.items()
                }

            else:
                raise RuntimeError(f"Unsupported planner {self._planner}")

        # Write metrics JSON to experiments directory
        self._write_metrics_to_disk()
