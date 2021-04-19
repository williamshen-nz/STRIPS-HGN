import logging
from typing import Dict, List, Tuple, Type

from torch.utils.data import DataLoader

from strips_hgn.config import (
    DEFAULT_DOMAIN_TO_MIN_SAMPLES,
    DOMAIN_TO_NUM_BINS_OVERRIDE,
)
from strips_hgn.features import (
    GlobalFeatureMapper,
    HyperedgeFeatureMapper,
    NodeFeatureMapper,
)
from strips_hgn.hypergraph.delete_relaxation import (
    DeleteRelaxationHypergraphView,
)
from strips_hgn.planning import STRIPSProblem
from strips_hgn.torch_utils.dataloaders import (
    HypergraphsTupleDataset,
    create_dataloaders,
)
from strips_hgn.training_data import TrainingPair
from strips_hgn.training_data.generate import (
    generate_optimal_state_value_pairs,
)
from strips_hgn.training_data.merge import merge_state_value_pairs_by_domain
from strips_hgn.training_data.process import get_kfold_training_data
from strips_hgn.training_data.save import save_training_data
from strips_hgn.utils.timer import timed
from strips_hgn.workflows.training_data import BaseTrainingDataWorkflow

_log = logging.getLogger(__name__)


class KFoldTrainingDataWorkflow(BaseTrainingDataWorkflow):
    """
    Workflow to generate k-fold training data, and post-process it into
    PyTorch objects given the set of training problems.
    """

    def __init__(
        self,
        problems: List[STRIPSProblem],
        batch_size: int,
        num_folds: int,
        num_bins: int,
        remove_duplicates: bool,
        shuffle: bool,
        global_feature_mapper_cls: Type[GlobalFeatureMapper],
        node_feature_mapper_cls: Type[NodeFeatureMapper],
        hyperedge_feature_mapper_cls: Type[HyperedgeFeatureMapper],
        experiment_dir: str,
    ):
        """
        Parameters
        ----------
        problems: training problems
        batch_size: batch size for torch Dataloaders
        num_folds: number of folds to use in k-fold
        num_bins: number of bins to use for heuristic binning
        remove_duplicates: whether to remove duplicates from training data
        shuffle: whether to shuffle the dataset
        global_feature_mapper_cls: the global feature mapper
        node_feature_mapper_cls: the feature mapper for propositions
        hyperedge_feature_mapper_cls: the feature mapper for actions
        experiment_dir: directory to store experiment results in
        """
        super().__init__(
            problems=problems,
            global_feature_mapper_cls=global_feature_mapper_cls,
            node_feature_mapper_cls=node_feature_mapper_cls,
            hyperedge_feature_mapper_cls=hyperedge_feature_mapper_cls,
            experiment_dir=experiment_dir,
        )

        self._batch_size = batch_size

        # k-fold training data related variables
        self._num_folds = num_folds
        self._num_bins = num_bins
        self._remove_duplicates = remove_duplicates
        self._shuffle = shuffle

    @timed("KFoldTrainingDataWorkflow.GenerateTrainingDataTime")
    def _generate_kfold_training_data(
        self,
    ) -> List[Tuple[List[TrainingPair], List[TrainingPair]]]:
        """
        Run the workflow for generating and processing training data into
        the form we expect.

        Returns
        -------
        List of length num_folds, with each element containing a tuple where
        the 1st element of the tuple is the training state-value pairs,
        and the 2nd element of the tuple is the validation state-value pairs
        for the given fold.
        """
        # 1. Generate optimal state-value pairs for all problems
        problem_to_state_value_pairs = generate_optimal_state_value_pairs(
            self._problems
        )

        # 2. Merge state-value pairs by domain
        domain_to_training_pairs = merge_state_value_pairs_by_domain(
            problem_to_state_value_pairs,
            remove_duplicates=self._remove_duplicates,
        )

        # 3. Get k-fold training data and resample if required
        kfold_training_data = get_kfold_training_data(
            domain_to_training_pairs=domain_to_training_pairs,
            num_folds=self._num_folds,
            num_bins=self._num_bins,
            domain_to_num_bins=DOMAIN_TO_NUM_BINS_OVERRIDE,
            domain_to_min_samples=DEFAULT_DOMAIN_TO_MIN_SAMPLES,
            shuffle=self._shuffle,
        )

        # 4. Save training data to the experiments directory
        save_training_data(
            self._experiments_dir,
            domain_to_training_pairs,
            kfold_training_data,
            indent=2,
        )
        return kfold_training_data

    @timed("KFoldTrainingDataWorkflow.ConvertToHypergraphsTupleTime")
    def _generate_hypergraphs_tuple_datasets(
        self,
        kfold_training_data: List[
            Tuple[List[TrainingPair], List[TrainingPair]]
        ],
    ) -> List[Tuple[HypergraphsTupleDataset, HypergraphsTupleDataset]]:
        """
        Converts the k-fold training data which is a list of length
        `self._num_folds`, where each element is a tuple containing the
        training set and validation set for the fold, respectively.

        Parameters
        ----------
        kfold_training_data

        Returns
        -------
        A list of length `self._num_folds`, where each element is a tuple
        containing the HypergraphsTupleDataset for the training and
        validation set, respectively.
        """
        # Generate the delete-relaxation hypergraph view of the problems
        problem_to_delete_relaxation_hypergraph: Dict[
            STRIPSProblem, DeleteRelaxationHypergraphView
        ] = {
            problem: DeleteRelaxationHypergraphView(problem)
            for problem in self._problems
        }

        kfold_hypergraphs_tuples = [
            (
                # Create the input and target HypergraphsTuple for the training
                # pairs in the training set of the current fold
                [
                    self._create_input_and_target_hypergraphs_tuple(
                        training_pair,
                        problem_to_delete_relaxation_hypergraph[
                            training_pair.problem
                        ],
                    )
                    for training_pair in train_set
                ],
                # Create the input and target HypergraphsTuple for the training
                # pairs in the validation set of the current fold
                [
                    self._create_input_and_target_hypergraphs_tuple(
                        training_pair,
                        problem_to_delete_relaxation_hypergraph[
                            training_pair.problem
                        ],
                    )
                    for training_pair in val_set
                ],
            )
            for train_set, val_set in kfold_training_data
        ]

        # Check number of folds maintained
        assert len(kfold_hypergraphs_tuples) == len(kfold_training_data)
        # Check size of training and validation set maintained
        for idx in (0, 1):
            assert [len(tup[idx]) for tup in kfold_hypergraphs_tuples] == [
                len(tup[idx]) for tup in kfold_training_data
            ]

        # Convert List[Tuple[List[HypergraphsTuple], List[HypergraphsTuple]]
        # to List[Tuple[HypergraphsTupleDataset, HypergraphsTupleDataset]]
        kfold_datasets = [
            (
                # Training dataset
                HypergraphsTupleDataset(*list(zip(*train_h_tuples))),
                # Validation dataset
                HypergraphsTupleDataset(*list(zip(*val_h_tuples))),
            )
            for train_h_tuples, val_h_tuples in kfold_hypergraphs_tuples
        ]

        _log.info(
            "Successfully created HypergraphsTupleDataset for training data."
        )
        return kfold_datasets

    @timed("KFoldTrainingDataWorkflow.TotalTime")
    def run(self) -> List[Tuple[DataLoader, DataLoader]]:
        # Generate optimal training data
        kfold_training_data = self._generate_kfold_training_data()

        # Convert to HypergraphsTupleTrainingPairs
        kfold_datasets = self._generate_hypergraphs_tuple_datasets(
            kfold_training_data
        )

        # Convert to torch DataLoaders
        kfold_dataloaders = create_dataloaders(
            kfold_datasets, batch_size=self._batch_size
        )
        return kfold_dataloaders
