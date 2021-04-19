from abc import ABC, abstractmethod
from typing import List

from strips_hgn.hypergraph import Hyperedge, Node
from strips_hgn.hypergraph.hypergraph_view import HypergraphView
from strips_hgn.utils import Number


class AbstractFeatureMapper(ABC):
    @classmethod
    @abstractmethod
    def name(cls) -> str:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def input_size(cls) -> int:
        """ Size of the feature vector returned by this feature mapper """
        raise NotImplementedError


class GlobalFeatureMapper(AbstractFeatureMapper, ABC):
    @abstractmethod
    def hypergraph_view_to_feature(
        self, hypergraph_view: HypergraphView
    ) -> List[Number]:
        raise NotImplementedError

    def __call__(self, hypergraph_view: HypergraphView) -> List[Number]:
        return self.hypergraph_view_to_feature(hypergraph_view)


class NodeFeatureMapper(AbstractFeatureMapper, ABC):
    @abstractmethod
    def node_to_feature(self, node: Node) -> List[Number]:
        raise NotImplementedError

    def __call__(self, node: Node) -> List[Number]:
        return self.node_to_feature(node)


class HyperedgeFeatureMapper(AbstractFeatureMapper, ABC):
    @abstractmethod
    def hyperedge_to_feature(self, hyperedge: Hyperedge) -> List[Number]:
        raise NotImplementedError

    def __call__(self, hyperedge: Hyperedge) -> List[Number]:
        return self.hyperedge_to_feature(hyperedge)
