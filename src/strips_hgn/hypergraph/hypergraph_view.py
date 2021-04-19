from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List

from strips_hgn.hypergraph import Hyperedge, Node
from strips_hgn.planning import STRIPSProblem
from strips_hgn.utils import Number

if TYPE_CHECKING:
    # To avoid circular imports for the type hinting
    from strips_hgn.features import (
        HyperedgeFeatureMapper,
        NodeFeatureMapper,
        GlobalFeatureMapper,
    )


class HypergraphView(ABC):
    """ Hypergraph View of a planning problem """

    def __init__(self, problem: STRIPSProblem):
        self.problem = problem

    def global_features(
        self, global_feature_mapper: "GlobalFeatureMapper"
    ) -> List[Number]:
        return global_feature_mapper(self)

    @property
    @abstractmethod
    def nodes(self) -> List[Node]:
        raise NotImplementedError

    @abstractmethod
    def node_to_idx(self, node: Node) -> int:
        raise NotImplementedError

    def node_features(
        self, node_feature_mapper: "NodeFeatureMapper"
    ) -> List[List[Number]]:
        """
        Maps the nodes in the hypergraph to their features using a feature
        mapper.

        Parameters
        ----------
        node_feature_mapper: NodeFeatureMapper

        Returns
        -------
        List[List[Number]], where each element represents the feature vector
        for the node at the given index.
        """
        return [node_feature_mapper(node) for node in self.nodes]

    @property
    @abstractmethod
    def hyperedges(self) -> List[Hyperedge]:
        raise NotImplementedError

    @abstractmethod
    def hyperedge_to_idx(self, hyperedge: Hyperedge) -> int:
        raise NotImplementedError

    def hyperedge_features(
        self, hyperedge_feature_mapper: "HyperedgeFeatureMapper"
    ) -> List[List[Number]]:
        """
        Maps the hyperedges in the hypergraph to their features using a feature
        mapper.

        Parameters
        ----------
        hyperedge_feature_mapper: HyperedgeFeatureMapper

        Returns
        -------
        List[List[Number]], where each element represents the feature vector
        for the hyperedge at the given index.
        """
        return [hyperedge_feature_mapper(edge) for edge in self.hyperedges]
