from typing import FrozenSet, Type

import torch

from hypergraph_nets.hypergraphs import HypergraphsTuple
from strips_hgn.features import (
    GlobalFeatureMapper,
    HyperedgeFeatureMapper,
    NodeFeatureMapper,
)
from strips_hgn.features.node_features import PropositionInStateAndGoal
from strips_hgn.hypergraph.hypergraph_view import HypergraphView
from strips_hgn.models.hypergraph_nets_adaptor import (
    hypergraph_view_to_hypergraphs_tuple,
)
from strips_hgn.planning import Proposition, STRIPSProblem


class BaseFeatureMappingWorkflow(object):
    """ Base Workflow which maps features """

    def __init__(
        self,
        global_feature_mapper_cls: Type[GlobalFeatureMapper],
        node_feature_mapper_cls: Type[NodeFeatureMapper],
        hyperedge_feature_mapper_cls: Type[HyperedgeFeatureMapper],
        max_receivers: int,
        max_senders: int,
    ):
        # Feature mappers
        self._global_feature_mapper_cls = global_feature_mapper_cls
        self._node_feature_mapper_cls = node_feature_mapper_cls
        self._hyperedge_feature_mapper_cls = hyperedge_feature_mapper_cls

        # Global feature mappers do not require context to anything, so we can
        # use a static object
        self._static_global_feature_mapper = global_feature_mapper_cls()

        # Hyperedge feature mappers do not require context to the current state
        # (at least for now), so we can use the same mapper
        self._static_hyperedge_feature_mapper = hyperedge_feature_mapper_cls()

        # Max receivers and senders
        self.max_receivers = max_receivers
        self.max_senders = max_senders

    def _get_global_feature_mapper(self) -> GlobalFeatureMapper:
        """ Get the Global feature mapper """
        return self._static_global_feature_mapper

    def _get_hyperedge_feature_mapper(
        self, problem: STRIPSProblem
    ) -> HyperedgeFeatureMapper:
        """
        Get the Hyperedge feature mapper for a STRIPS problem. For now, they do
        not require context to the current state, so we can just use the same
        static mapper.

        Parameters
        ----------
        problem: the STRIPS problem

        Returns
        -------
        HyperedgeFeatureMapper
        """
        return self._static_hyperedge_feature_mapper

    def _get_node_feature_mapper(
        self, current_state: FrozenSet[Proposition], problem: STRIPSProblem
    ) -> NodeFeatureMapper:
        """
        The node feature mappers need to be instantiated based on the current
        state and goal states. Hence, a separate one is needed for each
        state and planning problem

        Parameters
        ----------
        current_state: the current state
        problem: the STRIPS problem

        Returns
        -------
        NodeFeatureMapper
        """
        if self._node_feature_mapper_cls == PropositionInStateAndGoal:
            # Create node feature mapper for current state and the goal
            self._node_feature_mapper_cls: Type[PropositionInStateAndGoal]
            return self._node_feature_mapper_cls(
                current_state=current_state, goal_state=problem.goals
            )
        else:
            raise RuntimeError(
                f"Unsupported node feature mapper "
                f"{self._node_feature_mapper_cls}"
            )

    def _get_input_hypergraphs_tuple(
        self, current_state: FrozenSet[Proposition], hypergraph: HypergraphView
    ) -> HypergraphsTuple:
        """
        Computes and returns the input HypergraphsTuple for a state and a
        hypergraph view of the planning problem with its:
        
          - Node features
          - Hyperedge features
          - *NO* global features as we don't support them at the moment

        Parameters
        ----------
        current_state: the current state
        hypergraph: view of the hypergraph

        Returns
        -------
        HypergraphsTuple
        """
        # Get the global features and reshape so its shape is 1 x n
        global_features = hypergraph.global_features(
            self._get_global_feature_mapper()
        )
        global_features = (
            torch.tensor(global_features, dtype=torch.float32).reshape(1, -1)
            if global_features
            else None
        )

        return hypergraph_view_to_hypergraphs_tuple(
            hypergraph=hypergraph,
            receiver_k=self.max_receivers,
            sender_k=self.max_senders,
            # Map the nodes to their features
            node_features=torch.tensor(
                hypergraph.node_features(
                    self._get_node_feature_mapper(
                        current_state, hypergraph.problem
                    )
                ),
                dtype=torch.float32,
            ),
            # Map the hyperedges to their features
            edge_features=torch.tensor(
                hypergraph.hyperedge_features(
                    self._get_hyperedge_feature_mapper(hypergraph.problem)
                ),
                dtype=torch.float32,
            ),
            # Map the hypergraph to its global features
            global_features=global_features,
        )
