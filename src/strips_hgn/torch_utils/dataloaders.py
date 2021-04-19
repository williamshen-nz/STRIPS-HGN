import logging
from typing import List, NamedTuple, Tuple

from torch.utils.data import (
    DataLoader,
    Dataset,
    RandomSampler,
    SequentialSampler,
)

from hypergraph_nets.hypergraphs import HypergraphsTuple
from strips_hgn.models.hypergraph_nets_adaptor import merge_hypergraphs_tuple

_log = logging.getLogger(__name__)


class HypergraphsTupleTrainingPair(NamedTuple):
    """
    TrainingPair which consists of the input and target HypergraphsTuple
    """

    input_h_tuple: HypergraphsTuple
    target_h_tuple: HypergraphsTuple


class HypergraphsTupleDataset(Dataset):
    """ Wrapper for several input and target HypergraphsTuple pairs """

    def __init__(
        self,
        input_h_tuples: List[HypergraphsTuple],
        target_h_tuples: List[HypergraphsTuple],
    ):
        assert len(input_h_tuples) == len(target_h_tuples)
        self._input_h_tuples = input_h_tuples
        self._target_h_tuples = target_h_tuples

    def __len__(self) -> int:
        return len(self._input_h_tuples)

    def __getitem__(self, idx) -> Tuple[HypergraphsTuple, HypergraphsTuple]:
        return self._input_h_tuples[idx], self._target_h_tuples[idx]


def _collate_hypergraphs_tuples(
    h_tuples: List[Tuple[HypergraphsTuple, HypergraphsTuple]]
) -> Tuple[HypergraphsTuple, HypergraphsTuple]:
    """
    Collate multiple HypergraphsTuple into a single one.

    Parameters
    ----------
    h_tuples: a list of tuples containing the training and target
        HypergraphsTuple, respectively

    Returns
    -------
    A tuple with HypergraphsTuple, where the 1st element is the merged training
    hypergraphs, and the 2nd element is the merged target hypergraphs
    """
    return (
        # Merge input HypergraphsTuple
        merge_hypergraphs_tuple([tup[0] for tup in h_tuples]),
        # Merge target HypergraphsTuple
        merge_hypergraphs_tuple([tup[1] for tup in h_tuples]),
    )


def create_dataloaders(
    kfold_datasets: List[
        Tuple[HypergraphsTupleDataset, HypergraphsTupleDataset]
    ],
    batch_size: int = 1,
) -> List[Tuple[DataLoader, DataLoader]]:
    """
    Create the PyTorch dataloaders given the k-fold datasets

    Parameters
    ----------
    kfold_datasets: a list of tuples containing the training and validation
        HypergraphsTupleDataSet, where each element in the list contains the
        datasets for the fold.
    batch_size: batch size for training

    Returns
    -------
    List[Tuple[DataLoader, DataLoader]], with each element containing a
    tuple with the training dataloader and validation dataloader for the fold,
    respectively.
    """
    if batch_size < 1:
        raise ValueError("Batch size must be >= 1")

    # Create dataloaders for each fold
    kfold_dataloaders = []

    for train_dataset, val_dataset in kfold_datasets:
        kfold_dataloaders.append(
            (
                # Training dataloader
                DataLoader(
                    train_dataset,
                    sampler=RandomSampler(train_dataset, replacement=False),
                    batch_size=batch_size,
                    collate_fn=_collate_hypergraphs_tuples,
                ),
                # Validation dataloader, no need for random sampling
                DataLoader(
                    val_dataset,
                    sampler=SequentialSampler(val_dataset),
                    batch_size=batch_size,
                    collate_fn=_collate_hypergraphs_tuples,
                ),
            )
        )

    assert len(kfold_dataloaders) == len(kfold_datasets)
    return kfold_dataloaders
