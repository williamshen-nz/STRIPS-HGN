import pytest

from strips_hgn.features.hyperedge_features import (
    HyperedgeWeightOnly,
    ComplexHyperedgeFeatureMapper,
    MoreComplexHyperedgeFeatureMapper,
)
from strips_hgn.hypergraph import Hyperedge


@pytest.mark.parametrize(
    "hyperedge, expected_features",
    [
        (
            Hyperedge(
                "test_1", 5.0, frozenset({"a", "b"}), frozenset({"c", "d"})
            ),
            [5.0],
        ),
        (
            Hyperedge(
                "test_2", 3.14159, frozenset({"1", "2", "3"}), frozenset()
            ),
            [3.14159],
        ),
    ],
)
def test_hyperedge_weight_only(hyperedge, expected_features):
    feature_mapper = HyperedgeWeightOnly()
    assert feature_mapper(hyperedge) == expected_features


@pytest.mark.parametrize(
    "hyperedge, expected_features",
    [
        (
            Hyperedge(
                "edge_1", 120.5, frozenset({"a", "b"}), frozenset({"c", "d"})
            ),
            [120.5, 2, 2],
        ),
        (
            Hyperedge("edge_2", 42, frozenset({"1", "2", "3"}), frozenset()),
            [42, 3, 0],
        ),
        (
            Hyperedge("edge_3", 1.0, frozenset(), frozenset({"a", "b"})),
            [1.0, 0, 2],
        ),
    ],
)
def test_complex_hyperedge_feature_mapper(hyperedge, expected_features):
    feature_mapper = ComplexHyperedgeFeatureMapper()
    assert feature_mapper(hyperedge) == expected_features


@pytest.mark.parametrize(
    "hyperedge, expected_features",
    [
        (
            Hyperedge(
                "edge_1",
                120.5,
                frozenset({"a", "b"}),
                frozenset({"c", "d"}),
                context={"delete_effects": frozenset({"b"})},
            ),
            [120.5, 2, 2, 1],
        ),
        (
            Hyperedge(
                "edge_2",
                42,
                frozenset({"1", "2", "3"}),
                frozenset(),
                context={"delete_effects": frozenset({"4", "5", "6"})},
            ),
            [42, 3, 0, 3],
        ),
        (
            Hyperedge(
                "edge_3",
                1.0,
                frozenset(),
                frozenset({"a", "b"}),
                context={"delete_effects": frozenset()},
            ),
            [1.0, 0, 2, 0],
        ),
    ],
)
def test_more_complex_hyperedge_feature_mapper(hyperedge, expected_features):
    feature_mapper = MoreComplexHyperedgeFeatureMapper()
    assert feature_mapper(hyperedge) == expected_features


@pytest.mark.parametrize(
    "hyperedge",
    [
        # No context - i.e. None
        Hyperedge(
            "my_edge_1", 1.0, frozenset({"a", "b", "c"}), frozenset({"d", "e"})
        ),
        # Has context but does not contain delete effects
        Hyperedge(
            "my_edge_2",
            420,
            frozenset({"1", "2"}),
            frozenset({"3", "4", "5", "6"}),
            context={"my_cool_context": 0},
        ),
    ],
)
def test_more_complex_hyperedge_feature_mapper_without_context(hyperedge):
    feature_mapper = MoreComplexHyperedgeFeatureMapper()
    with pytest.raises(ValueError):
        feature_mapper(hyperedge)
