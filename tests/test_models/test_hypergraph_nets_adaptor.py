import os

import pytest
import torch

from hypergraph_nets.hypergraphs import HypergraphsTuple
from strips_hgn.features.global_features import NumberOfNodesAndEdgesGlobalFeatureMapper
from strips_hgn.features.hyperedge_features import ComplexHyperedgeFeatureMapper
from strips_hgn.features.node_features import PropositionInStateAndGoal
from strips_hgn.hypergraph.delete_relaxation import DeleteRelaxationHypergraphView
from strips_hgn.models.hypergraph_nets_adaptor import merge_hypergraphs_tuple
from strips_hgn.planning.strips import _PyperplanSTRIPSProblem
from strips_hgn.workflows.base_workflow import BaseFeatureMappingWorkflow

_module_path = os.path.dirname(os.path.abspath(__file__))

_domain_path = os.path.join(_module_path, "assets/domain.pddl")
_problem_path = os.path.join(_module_path, "assets/problem.pddl")


@pytest.fixture
def hg_tuple() -> HypergraphsTuple:
    max_num_add_effects = 10
    max_num_preconditions = 10

    # Setup STRIPS-HGN so we can get a hypergraph tuple of the initial state
    strips_problem = _PyperplanSTRIPSProblem(_domain_path, _problem_path)
    problem_dr_hypergraph = DeleteRelaxationHypergraphView(strips_problem)
    state_hypergraph_encoder = BaseFeatureMappingWorkflow(
        global_feature_mapper_cls=NumberOfNodesAndEdgesGlobalFeatureMapper,
        node_feature_mapper_cls=PropositionInStateAndGoal,
        hyperedge_feature_mapper_cls=ComplexHyperedgeFeatureMapper,
        max_receivers=max_num_add_effects,
        max_senders=max_num_preconditions,
    )

    hg_tuple = state_hypergraph_encoder._get_input_hypergraphs_tuple(
        current_state=strips_problem.initial_state, hypergraph=problem_dr_hypergraph
    )
    yield hg_tuple


def test_merge_hypergraphs_tuple_single_element(hg_tuple: HypergraphsTuple):
    merged_hg_tuple = merge_hypergraphs_tuple([hg_tuple])
    assert torch.equal(merged_hg_tuple.edges, hg_tuple.edges)
    assert torch.equal(merged_hg_tuple.nodes, hg_tuple.nodes)
    assert torch.equal(merged_hg_tuple.globals, hg_tuple.globals)
    assert torch.equal(merged_hg_tuple.receivers, hg_tuple.receivers)
    assert torch.equal(merged_hg_tuple.senders, hg_tuple.senders)
    assert torch.equal(merged_hg_tuple.n_node, hg_tuple.n_node)
    assert torch.equal(merged_hg_tuple.n_edge, hg_tuple.n_edge)


def test_merge_hypergraphs_tuple_duplicate(hg_tuple: HypergraphsTuple):
    """Test duplicating the same hypergraph tuple"""
    # Effectively duplicate the hypergraph tuple
    merged_hg_tuple = merge_hypergraphs_tuple([hg_tuple, hg_tuple])
    assert merged_hg_tuple.total_n_node == 2 * hg_tuple.total_n_node
    assert merged_hg_tuple.total_n_edge == 2 * hg_tuple.total_n_edge

    # Check the node indices have been accumulated correctly (i.e., we do not use the same indices for the two merged
    # hypergraphs). There are twice as many nodes minus 1 to account for the -1 index used for padding (IIRC).
    og_node_idxs = set(
        hg_tuple.receivers.flatten().tolist() + hg_tuple.senders.flatten().tolist()
    )
    merged_node_idxs = set(
        merged_hg_tuple.receivers.flatten().tolist()
        + merged_hg_tuple.senders.flatten().tolist()
    )
    assert len(merged_node_idxs) == 2 * len(og_node_idxs) - 1

    # Check the node indices in the hyperedges have been accumulated correctly
    duplicated_hg_receivers = merged_hg_tuple.receivers[hg_tuple.total_n_edge :]
    duplicated_hg_senders = merged_hg_tuple.senders[hg_tuple.total_n_edge :]
    assert (
        len(duplicated_hg_receivers)
        == len(duplicated_hg_senders)
        == hg_tuple.total_n_edge
    )

    # Compute expected indices and check, need to make sure we maintain -1 padding
    expected_hg_receivers = hg_tuple.receivers
    expected_hg_receivers[hg_tuple.receivers != -1] += hg_tuple.total_n_node
    assert torch.equal(duplicated_hg_receivers, expected_hg_receivers)

    expected_hg_senders = hg_tuple.senders
    expected_hg_senders[hg_tuple.senders != -1] += hg_tuple.total_n_node
    assert torch.equal(duplicated_hg_senders, expected_hg_senders)
