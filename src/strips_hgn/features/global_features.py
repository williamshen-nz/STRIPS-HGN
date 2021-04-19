from typing import List, Type

from strips_hgn.features import GlobalFeatureMapper
from strips_hgn.hypergraph.hypergraph_view import HypergraphView
from strips_hgn.utils import Number


class EmptyGlobalFeatureMapper(GlobalFeatureMapper):
    """ Global Feature Mapper that does nothing """

    def hypergraph_view_to_feature(
        self, hypergraph_view: HypergraphView
    ) -> List[Number]:
        return []

    @classmethod
    def name(cls) -> str:
        return "none"

    @classmethod
    def input_size(cls) -> int:
        return 0


class NumberOfNodesAndEdgesGlobalFeatureMapper(GlobalFeatureMapper):
    """
    Map a Hypergraph View to a feature vector containing the number of nodes
    and edges in the hypergraph.

    [number of nodes, number of hyperedges]

    For STRIPS problems, this corresponds to the number of propositions and
    grounded actions, respectively.
    """

    def hypergraph_view_to_feature(
        self, hypergraph_view: HypergraphView
    ) -> List[Number]:
        return [len(hypergraph_view.nodes), len(hypergraph_view.hyperedges)]

    @classmethod
    def name(cls) -> str:
        return "num_nodes_and_edges"

    @classmethod
    def input_size(cls) -> int:
        return 2


# Used for command line args
GLOBAL_FEATURE_MAPPERS = {
    feature_mapper_cls.name(): feature_mapper_cls
    for feature_mapper_cls in (
        EmptyGlobalFeatureMapper,
        NumberOfNodesAndEdgesGlobalFeatureMapper,
    )
}

DEFAULT_GLOBAL_FEATURE_MAPPER = EmptyGlobalFeatureMapper.name()
assert DEFAULT_GLOBAL_FEATURE_MAPPER in GLOBAL_FEATURE_MAPPERS


def get_global_feature_mapper(mapper: str) -> Type[GlobalFeatureMapper]:
    if mapper not in GLOBAL_FEATURE_MAPPERS:
        raise ValueError(f"Unsupported global feature mapper {mapper}")

    return GLOBAL_FEATURE_MAPPERS[mapper]
