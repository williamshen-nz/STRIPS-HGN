import logging
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.utils import resample

from strips_hgn.config import TRAINING_DATA_TIMER_LOG_LEVEL
from strips_hgn.training_data import TrainingPair
from strips_hgn.utils import Number
from strips_hgn.utils.metrics import CountMetric, metrics_logger
from strips_hgn.utils.timer import timed

_log = logging.getLogger(__name__)


@timed("GetKFoldTrainingDataTime", log_level=TRAINING_DATA_TIMER_LOG_LEVEL)
def get_kfold_training_data(
    domain_to_training_pairs: Dict[str, List[TrainingPair]],
    num_folds: int,
    num_bins: int = -1,
    domain_to_num_bins: Dict[str, int] = None,
    domain_to_min_samples: Dict[str, int] = None,
    shuffle: bool = True,
) -> List[Tuple[List[TrainingPair], List[TrainingPair]]]:
    """
    Applies K-Fold to get training and validation sets.
    Training data from multiple domains are merged such that each fold contains
    data from each considered domain.

    Also performs stratified sampling with replacement if the number of samples
    for a domain is less than the minimum number of samples.

    Parameters
    ----------
    domain_to_training_pairs: mapping of a domain name to a list of training
        pairs
    num_folds: number of folds, i.e. k in the paper
    num_bins: number of bins to split data into by target heuristic value
    domain_to_num_bins: mapping of a Domain name to the number of bins to use
        for each domain. This will override the 'num_bins' for any domains
        specified in the dict.
    domain_to_min_samples: mapping of a Domain name to the minimum to number
        of training samples for the domain. If a domain is not specified
        the minimum number of samples is assumed to be 0.
    shuffle: whether to shuffle the training data before splitting into
        folds and bins

    Returns
    -------
    List[Tuple[List[TrainingPair], List[TrainingPair]]]
        a list of length 'num_folds', with each element representing a fold
        containing tuples with (training pairs, validation pairs)
    """
    if num_folds < 2:
        raise ValueError("k >= 2 folds in order to apply stratified k-fold")
    if not (num_bins >= 1 or domain_to_num_bins):
        raise ValueError(
            "There must be at least one bin, or 'domain_to_num_bins' must be "
            "specified"
        )

    total_num_pairs = 0
    # total_num_pairs_processed includes the pairs processed for each fold
    total_num_pairs_processed = 0

    # Stratified k-fold and mapping of fold to training pairs
    skf = StratifiedKFold(n_splits=num_folds, shuffle=shuffle)
    fold_idx_to_training_pairs = defaultdict(list)
    fold_idx_to_validation_pairs = defaultdict(list)

    # Apply quantile binning for training data in each domain and split into
    # the k-folds using the bins
    for domain, training_pairs in domain_to_training_pairs.items():
        _log.info(f"Processing training data for '{domain}'")
        heuristic_values = [pair.value for pair in training_pairs]

        # Number of bins for this domain
        if domain_to_num_bins and domain in domain_to_num_bins:
            num_bins_for_domain = domain_to_num_bins[domain]
            _log.warning(
                f"Number of bins for '{domain}' overriden to "
                f"{num_bins_for_domain}"
            )
        else:
            num_bins_for_domain = num_bins

        # Bin the target values
        bin_idx = _get_bins(num_bins_for_domain, heuristic_values)
        assert len(training_pairs) == len(heuristic_values) == len(bin_idx)

        min_samples = (
            domain_to_min_samples.get(domain, 0)
            if domain_to_min_samples
            else 0
        )

        # Resample using stratified sampling with replacement if required, by
        # using the binned heuristic value.
        resampled = False
        if len(training_pairs) < min_samples:
            _log.warning(
                f"Number of samples {len(training_pairs)} found for "
                f"{domain}, less than min required samples = {min_samples}. "
                "Resampling using initial heuristic bins."
            )
            training_pairs, heuristic_values = resample(
                training_pairs,
                heuristic_values,
                n_samples=min_samples,
                stratify=bin_idx,
            )
            assert len(training_pairs) == len(heuristic_values) == min_samples

            # Refit the bins
            _log.debug(
                "Refitting heuristic value bins since we resampled "
                f"training pairs to {min_samples} samples"
            )
            bin_idx = _get_bins(num_bins_for_domain, heuristic_values)
            resampled = True

        # Final number of training pairs after resampling
        metrics_logger.add_metric(
            CountMetric(
                "FinalNumberOfTrainingPairs",
                len(training_pairs),
                context={"domain": domain},
            )
        )
        total_num_pairs += len(training_pairs)

        # Perform the stratified k-fold split
        num_pairs_processed = 0
        for fold_idx, (train_idx, val_idx) in enumerate(
            skf.split(training_pairs, bin_idx)
        ):
            # Add training and validation training pairs for this fold
            fold_idx_to_training_pairs[fold_idx].extend(
                training_pairs[idx] for idx in train_idx
            )
            fold_idx_to_validation_pairs[fold_idx].extend(
                training_pairs[idx] for idx in val_idx
            )
            num_pairs_processed += len(train_idx) + len(val_idx)

        # Check we have processed expected number of training pairs
        if resampled:
            assert np.isclose(num_pairs_processed, min_samples * num_folds)
        else:
            assert np.isclose(
                num_pairs_processed, len(training_pairs) * num_folds
            )
        metrics_logger.add_metric(
            CountMetric(
                "NumberOfTrainingPairsProcessed",
                num_pairs_processed,
                context={"operation": "StratifiedKFold", "domain": domain},
            )
        )
        total_num_pairs_processed += num_pairs_processed

    # Merge training and validation pairs
    kfold_training_data = []
    for fold_idx in range(num_folds):
        kfold_training_data.append(
            (
                fold_idx_to_training_pairs[fold_idx],
                fold_idx_to_validation_pairs[fold_idx],
            )
        )
    assert len(kfold_training_data) == num_folds
    _log.info("Finished generating k-fold training data")

    metrics_logger.add_metric(
        CountMetric("TotalFinalNumberOfTrainingPairs", total_num_pairs)
    )
    metrics_logger.add_metric(
        CountMetric(
            "TotalNumberOfTrainingPairsProcessed",
            total_num_pairs_processed,
            context={"operation": "StratifiedKFold"},
        )
    )

    return kfold_training_data


def _get_bins(num_bins: int, values: List[Number]):
    """
    Perform equal frequency binning (i.e. quantile binning) to bin the values
    into `num_bins` bins.

    Parameters
    ----------
    num_bins
    values

    Returns
    -------
    bin indices
    """
    # Turn values into np array or sklearn will complain
    np_values = np.array(values).reshape(-1, 1)

    # Equal frequency binning (i.e. quantile binning)
    binner = KBinsDiscretizer(
        n_bins=num_bins, encode="ordinal", strategy="quantile"
    )
    binner.fit(np_values)

    # Bin that each training pair belongs to
    bin_idx = binner.transform(np_values).reshape(-1)

    # Debug
    _log.debug(f"Split heuristic values into {num_bins} bins")
    _log.debug(f"Bin Edges: {binner.bin_edges_}")
    _log.debug(f"Bin Frequencies: {Counter(bin_idx.tolist())}")
    _log.debug(f"Heuristic Frequencies: {Counter(values)}")
    return bin_idx
