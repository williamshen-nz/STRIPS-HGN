import json
import logging
import os
import re
from datetime import datetime
from typing import Optional

from strips_hgn.utils.args.base_args import BaseArgs
from strips_hgn.utils.json_encoders import ArgsEncoder
from strips_hgn.utils.logging_setup import setup_full_logging

_log = logging.getLogger(__name__)

_EXPERIMENT_DIRECTORY_FORMAT = "strips-hgn-{datetime}"


def natural_sort_key(s, _nsre=re.compile("([0-9]+)")):
    """ Taken from: https://stackoverflow.com/a/16090640 """
    return [
        int(text) if text.isdigit() else text.lower()
        for text in _nsre.split(s)
    ]


def natural_sort(l):
    """ https://stackoverflow.com/a/4836734 """
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(l, key=alphanum_key)


class Namespace(dict):
    def __init__(self, **kwargs):
        super().__init__(kwargs)
        self.__dict__.update(kwargs)


def create_experiment_results_directory(
    prefix: str, base_results_dir: str, experiment_dir: Optional[str] = None
) -> str:
    """
    Create the results directory for an experiment

    Parameters
    ----------
    prefix: prefix to add to experiments dir
    base_results_dir: base directory for storing all results, e.g. '../results'
    experiment_dir: directory to store the results of the specific
        experiment, e.g. 'strips-hgn-blocksworld'

    Returns
    -------
    str, the directory where results of the experiment will be stored
    """
    if not experiment_dir:
        experiment_dir = "-".join(
            [
                prefix,
                _EXPERIMENT_DIRECTORY_FORMAT.format(
                    datetime=datetime.now().isoformat()
                ),
            ]
        )
    full_experiment_dir = os.path.join(base_results_dir, experiment_dir)

    if os.path.exists(full_experiment_dir):
        # Check directory doesn't already exist
        raise RuntimeError(
            f"Experiment directory {full_experiment_dir} already exists"
        )

    os.makedirs(full_experiment_dir)
    return full_experiment_dir


def setup_experiment(
    prefix: str, base_results_dir: str, log_level=logging.INFO
) -> str:
    """
    Sets up an experiment by creating directories and logger

    Parameters
    ----------
    prefix: prefix to add to the experiments dir
    base_results_dir: str, base directory which is used to store all results
    log_level

    Returns
    -------
    str, path to the experiment results directory
    """
    # Create directory to store experiment results
    full_experiment_dir = create_experiment_results_directory(
        prefix, base_results_dir
    )

    # Setup logger
    setup_full_logging(full_experiment_dir, log_level=log_level)

    # Log experiments directory after logger has been set up
    _log.info(f"Experiment results directory: {full_experiment_dir}")

    return full_experiment_dir


def dump_args(args: BaseArgs, results_dir: str, dump_fname: str) -> str:
    """
    Dump args to a JSON

    Parameters
    ----------
    args: the arguments created
    results_dir: directory to store JSON in
    dump_fname: name of JSON file

    Returns
    -------
    Filename of the arg dump in JSON
    """
    args_fname = os.path.join(results_dir, dump_fname)
    json.dump(args.__dict__, open(args_fname, "w"), indent=2, cls=ArgsEncoder)
    return args_fname
