import os
from dataclasses import dataclass
from typing import List

from strips_hgn.planning import (
    Heuristic,
    PlannerForEvaluation,
    SearchAlgorithm,
)
from strips_hgn.utils.args.base_args import BaseArgs, get_base_parser


@dataclass(frozen=True)
class EvaluationArgs(BaseArgs):
    """
    Fields
    ------
    checkpoint: the checkpoint file path which contains the STRIPS-HGN model
    heuristics: list of Heuristics to compare the STRIPS-HGN against
    search_algorithm: search algorithm to run for the analysis
    planner: planner to use for evaluation
    max_search_time: max search time for each problem, in seconds
    """

    checkpoint: str
    heuristics: List[Heuristic]
    search_algorithm: SearchAlgorithm
    planner: PlannerForEvaluation
    max_search_time: float

    def validate(self):
        super().validate()
        assert os.path.exists(
            self.checkpoint
        ), "Checkpoint file does not exist!"
        assert self.max_search_time > 0.0


def _get_evaluation_parser(show_defaults=True):
    """ Generate parser for evaluation scripts """
    parser = get_base_parser(
        "Learning Heuristics over Hypergraphs. Evaluation Script",
        show_defaults,
    )

    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        help="File path containing the checkpoint of the STRIPS-HGN for the "
        "learned heuristic",
        required=True,
    )

    parser.add_argument(
        "-H",
        "--heuristics",
        choices=Heuristic.member_names(),
        nargs="+",
        help="Heuristics to compare against (if any)",
        required=False,
        default=[],
    )

    parser.add_argument(
        "-s",
        "--search-algorithm",
        choices=SearchAlgorithm.member_names(),
        default=SearchAlgorithm.a_star.name,
        type=lambda val: SearchAlgorithm.from_str(val),
        help="Search algorithm to use",
    )

    parser.add_argument(
        "--planner",
        choices=PlannerForEvaluation.member_names(),
        default=PlannerForEvaluation.pyperplan.name,
        type=lambda val: PlannerForEvaluation.from_str(val),
        help="Planner to use for evaluation",
    )

    parser.add_argument(
        "-t",
        "--max-search-time",
        type=float,
        default=300.0,
        help="Maximum search time for each problem in seconds",
    )

    return parser


def parse_and_validate_evaluation_args() -> EvaluationArgs:
    """ Parse args, validate and return TrainingArgs object """
    args = _get_evaluation_parser().parse_args()
    # Override heuristics from str to objects
    args.heuristics = [Heuristic.from_str(val) for val in args.heuristics]

    args = EvaluationArgs(**vars(args))
    args.validate()
    return args
