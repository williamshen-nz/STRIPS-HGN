from typing import List, Type

from strips_hgn.features import HyperedgeFeatureMapper
from strips_hgn.hypergraph import Hyperedge
from strips_hgn.utils import Number


class HyperedgeWeightOnly(HyperedgeFeatureMapper):
    """
    Maps a hyperedge (action) to a 1-dimensional feature vector containing
    its weight only (i.e. cost of the action).
    """

    def hyperedge_to_feature(self, hyperedge: Hyperedge) -> List[Number]:
        return [hyperedge.weight]

    @classmethod
    def name(cls) -> str:
        return "weight-only"

    @classmethod
    def input_size(cls) -> int:
        return 1


class ComplexHyperedgeFeatureMapper(HyperedgeFeatureMapper):
    """
    Maps a hyperedge (action) to a 3-dimensional feature vector where:
      [weight, number of senders, number of receivers]

    For hyperedges in a delete-relaxed hypergraph, this is:
      [action cost, number of preconditions, number of positive effects]
    """

    def hyperedge_to_feature(self, hyperedge: Hyperedge) -> List[Number]:
        return [
            hyperedge.weight,
            len(hyperedge.senders),
            len(hyperedge.receivers),
        ]

    @classmethod
    def name(cls) -> str:
        return "complex"

    @classmethod
    def input_size(cls) -> int:
        return 3


class MoreComplexHyperedgeFeatureMapper(HyperedgeFeatureMapper):
    """
    Maps a hyperedge (action) to a 4-dimensional feature vector, which for
    hyperedges in a delete-relaxed hypergraph corresponds to:

      [
          action cost,
          number of preconditions,
          number of positive effects,
          number of delete effects,
      ]

    TODO: better naming for this class
    """

    def hyperedge_to_feature(self, hyperedge: Hyperedge) -> List[Number]:
        # Check we have access to the delete effects in the Hyperedge's context
        if (
            hyperedge.context is None
            or "delete_effects" not in hyperedge.context
        ):
            raise ValueError(
                "Delete Effects could not be found in hyperedge "
                f"'{hyperedge.name}' context"
            )

        return [
            hyperedge.weight,
            len(hyperedge.senders),
            len(hyperedge.receivers),
            len(hyperedge.context["delete_effects"]),
        ]

    @classmethod
    def name(cls) -> str:
        return "more-complex"

    @classmethod
    def input_size(cls) -> int:
        return 4


# Used for command line args
EDGE_FEATURE_MAPPERS = {
    feature_mapper_cls.name(): feature_mapper_cls
    for feature_mapper_cls in (
        HyperedgeWeightOnly,
        ComplexHyperedgeFeatureMapper,
        MoreComplexHyperedgeFeatureMapper,
    )
}

DEFAULT_EDGE_FEATURE_MAPPER = ComplexHyperedgeFeatureMapper.name()
assert DEFAULT_EDGE_FEATURE_MAPPER in EDGE_FEATURE_MAPPERS


def get_hyperedge_feature_mapper(mapper: str) -> Type[HyperedgeFeatureMapper]:
    if mapper not in EDGE_FEATURE_MAPPERS:
        raise ValueError(f"Unsupported hyperedge feature mapper {mapper}")

    return EDGE_FEATURE_MAPPERS[mapper]
