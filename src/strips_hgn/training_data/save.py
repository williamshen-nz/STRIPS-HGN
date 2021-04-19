import json
import logging
import os
from json import JSONEncoder
from typing import Dict, List, Tuple

from strips_hgn.training_data import StateValuePair, TrainingPair

TRAINING_DATA_TIMER_LOG_LEVEL = logging.DEBUG

_log = logging.getLogger(__name__)


class _TrainingDataEncoder(JSONEncoder):
    def default(self, o):
        if isinstance(o, StateValuePair) or isinstance(o, TrainingPair):
            return o.to_json_dict()
        else:
            super(_TrainingDataEncoder, self).default(o)


def save_training_data(
    experiments_dir: str,
    domain_to_training_pairs: Dict[str, List[TrainingPair]],
    kfold_training_data: List[Tuple[List[TrainingPair], List[TrainingPair]]],
    **json_kwargs,
):
    """
    Dump the training data to the experiments directory

    Parameters
    ----------
    experiments_dir
    domain_to_training_pairs
    kfold_training_data
    json_kwargs: kwargs to pass to json.dump

    Returns
    -------
    None
    """
    training_data_dir = os.path.join(experiments_dir, "training_data")
    os.makedirs(training_data_dir)

    # Restructure object so it's easier to read in JSON
    fold_idx_to_training_data = {
        f"fold_{fold_idx}": {
            "training": training_data[0],
            "validation": training_data[1],
        }
        for fold_idx, training_data in enumerate(kfold_training_data)
    }

    for obj, json_name in [
        (domain_to_training_pairs, "domain_to_training_pairs.json"),
        (fold_idx_to_training_data, "kfold_training_data.json"),
    ]:
        fname = os.path.join(training_data_dir, json_name)
        json.dump(
            obj, open(fname, "w"), cls=_TrainingDataEncoder, **json_kwargs
        )

    _log.info(f"Saved training data details to {training_data_dir}")
