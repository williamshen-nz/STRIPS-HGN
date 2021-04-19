import logging
from typing import Callable, Optional

from strips_hgn.utils.args.base_args import BaseArgs
from strips_hgn.utils.helpers import dump_args, setup_experiment
from strips_hgn.utils.metrics import metrics_logger

_log = logging.getLogger(__name__)


def wrap_method(
    args: BaseArgs,
    wrapped_method: Callable[[BaseArgs, str], None],
    experiment_type: str,
    results_directory: str,
    experiments_dir: Optional[str] = None,
):
    """
    This method is used to wrap main methods by clearing metrics, setting
    up logging, dumping args to JSON, and handling exceptions through the
    try...except, etc.

    Parameters
    ----------
    args: TrainingArgs
    wrapped_method: the method to call
    experiment_type: the type of experiment (e.g. train, eval)
    results_directory: base directory where results are stored
    experiments_dir: directory to store experimental results. If this is not
        specified, then we setup logging and create a new directory.

    Returns
    -------
    None
    """
    # Clear any existing metrics
    metrics_logger.clear()

    # Setup experiment result directory and logger
    if not experiments_dir:
        log_level = logging.DEBUG if args.debug else logging.INFO
        experiments_dir = setup_experiment(
            experiment_type, results_directory, log_level
        )

    dump_args(args, experiments_dir, f"{experiment_type}_args.json")

    # Dump metrics to disk if any error is detected
    exception = None
    try:
        # Run the wrapped method
        wrapped_method(args, experiments_dir)

    except Exception as ex:
        _log.error("An unexpected error occurred.", exc_info=True)
        exception = ex
    finally:
        metrics_logger.save_json(
            experiments_dir,
            file_name=f"{experiment_type}_metrics.json",
            indent=2,
        )

        # Re-raise exception as the logger will log it
        if exception:
            print(
                f"Check {experiments_dir}/strips_hgn.log if error traceback "
                "is not listed"
            )
            raise exception

        _log.info(f"{experiment_type} script complete!")
