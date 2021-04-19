import logging
from abc import ABC, abstractmethod
from typing import List, Tuple, Type

import torch

from hypergraph_nets.hypergraphs import HypergraphsTuple
from strips_hgn.features import (
    GlobalFeatureMapper,
    HyperedgeFeatureMapper,
    NodeFeatureMapper,
)
from strips_hgn.hypergraph.hypergraph_view import HypergraphView
from strips_hgn.models.hypergraph_nets_adaptor import (
    hypergraph_view_to_hypergraphs_tuple,
)
from strips_hgn.planning import STRIPSProblem
from strips_hgn.planning.utils import (
    max_number_of_add_effects,
    max_number_of_preconditions,
)
from strips_hgn.torch_utils.dataloaders import HypergraphsTupleTrainingPair
from strips_hgn.training_data import TrainingPair
from strips_hgn.workflows.base_workflow import BaseFeatureMappingWorkflow

_log = logging.getLogger(__name__)


class BaseTrainingDataWorkflow(BaseFeatureMappingWorkflow, ABC):
    """
    Base class for all Training Data Workflows which generate and post-process
    training data to PyTorch objects
    """

    def __init__(
        self,
        problems: List[STRIPSProblem],
        global_feature_mapper_cls: Type[GlobalFeatureMapper],
        node_feature_mapper_cls: Type[NodeFeatureMapper],
        hyperedge_feature_mapper_cls: Type[HyperedgeFeatureMapper],
        experiment_dir: str,
    ):
        super().__init__(
            global_feature_mapper_cls=global_feature_mapper_cls,
            node_feature_mapper_cls=node_feature_mapper_cls,
            hyperedge_feature_mapper_cls=hyperedge_feature_mapper_cls,
            max_receivers=max_number_of_add_effects(problems),
            max_senders=max_number_of_preconditions(problems),
        )
        # Log maximum number of receivers and senders
        _log.info(f"Max number of EFF+ (receivers) = {self.max_receivers}")
        _log.info(f"Max number of PREs (senders) = {self.max_senders}")

        self._problems = problems
        self._experiments_dir = experiment_dir

    def _create_input_and_target_hypergraphs_tuple(
        self, training_pair: TrainingPair, hypergraph: HypergraphView
    ) -> Tuple[HypergraphsTuple, HypergraphsTuple]:
        """
        Create the input and target HypergraphsTuple objects for a training
        pair and its associated hypergraph.

        Parameters
        ----------
        training_pair: Training Pair with state-value pair
        hypergraph: the HypergraphView of the problem associated with the
            training pair

        Returns
        -------
        Tuple containing the input and target HypergraphsTuple objects,
        respectively.
        """
        assert training_pair.problem == hypergraph.problem

        # The input HypergraphsTuple with its node and hyperedge features.
        input_h_tuple = self._get_input_hypergraphs_tuple(
            current_state=training_pair.state, hypergraph=hypergraph
        )

        # The target HypergraphsTuple has its global feature set to the target
        # heuristic value. It doesn't have any node/hyperedge features
        target_h_tuple = hypergraph_view_to_hypergraphs_tuple(
            hypergraph=hypergraph,
            receiver_k=self.max_receivers,
            sender_k=self.max_senders,
            global_features=torch.tensor(
                [training_pair.value], dtype=torch.float32
            ),
        )

        return HypergraphsTupleTrainingPair(input_h_tuple, target_h_tuple)

    @abstractmethod
    def run(self):
        raise NotImplementedError
