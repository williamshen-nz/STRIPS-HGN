import logging
import os
import sys

import coloredlogs

_log = logging.getLogger(__name__)


_LOG_LEVEL_WARNING_MODULES = ("fast_downward", "matplotlib", "tensorflow")


def setup_logging_config(log_level=logging.INFO, filename=None):
    """
    Setup basic logging config

    Parameters
    ----------
    log_level
    filename: str, path to file to save the logging output to (if any)

    Returns
    -------
    None
    """
    # Get root logger and clear existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers = []

    kwargs = {
        "logger": root_logger,
        "level": log_level,
        "fmt": "%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
        "datefmt": "%Y-%m-%d %H:%M:%S",
        "stream": sys.stdout,
    }

    # Enable coloured logging
    coloredlogs.install(**kwargs)

    # Add new FileHandler which logs will be written to if required
    if filename:
        # Setup log level and formatting because coloredlogs overrides handlers
        handler = logging.FileHandler(filename)
        handler.setLevel(log_level)
        handler.setFormatter(
            logging.Formatter(kwargs["fmt"], kwargs["datefmt"])
        )
        root_logger.addHandler(handler)

    # Set log level for different modules
    logging.getLogger("pyperplan").setLevel(logging.INFO)
    for module_str in _LOG_LEVEL_WARNING_MODULES:
        logging.getLogger(module_str).setLevel(logging.WARNING)

    # Lightning has some weird logging config that we need to override
    lightning_logger = logging.getLogger("lightning")
    lightning_logger.handlers = []


def setup_full_logging(
    full_experiment_dir: str,
    log_name: str = "strips_hgn.log",
    log_level=logging.INFO,
):
    """
    Setup logging

    Parameters
    ----------
    full_experiment_dir: str, directory where log is going to be stored
    log_name: name of the log file
    log_level

    Returns
    -------
    None
    """
    _log_fname = os.path.join(full_experiment_dir, log_name)
    setup_logging_config(filename=_log_fname, log_level=log_level)
    _log.info(f"Writing logs to {_log_fname}")
