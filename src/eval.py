#!/usr/bin/env python

import logging
import os

from strips_hgn.models.strips_hgn import STRIPSHGN
from strips_hgn.utils.args import (
    parse_and_validate_evaluation_args,
    EvaluationArgs,
)
from strips_hgn.utils.timer import timed
from strips_hgn.utils.wrapper import wrap_method
from strips_hgn.workflows import EvaluateSTRIPSHGNWorkflow

_log = logging.getLogger(__name__)

_RESULTS_DIRECTORY = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "results"
)


@timed("EvaluationDriverMethodTime")
def eval_main(args: EvaluationArgs, experiments_dir: str):
    """
    Main runner method for evaluating a STRIPS-HGN.

    Parameters
    ----------
    args: EvaluationArgs
    experiments_dir: directory where experiment results will be stored
    """
    # Load model and run the evaluation workflow
    model = STRIPSHGN.load_from_checkpoint(args.checkpoint)

    if args.debug:
        _log.info(f"STRIPS-HGN hparams: {model.hparams}")

    eval_wf = EvaluateSTRIPSHGNWorkflow(
        model=model,
        heuristics=args.heuristics,
        search_algorithm=args.search_algorithm,
        planner=args.planner,
        max_search_time=args.max_search_time,
        experiments_dir=experiments_dir,
    )
    eval_wf.run(problems=args.get_strips_problems())


def eval_wrapper(args: EvaluationArgs):
    # Wrap the evaluation method
    wrap_method(
        args=args,
        wrapped_method=eval_main,
        experiment_type="eval",
        results_directory=_RESULTS_DIRECTORY,
    )


if __name__ == "__main__":
    eval_wrapper(args=parse_and_validate_evaluation_args())
