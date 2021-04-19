import pytest

from strips_hgn.hypergraph.delete_relaxation import \
    DeleteRelaxationHypergraphView
from strips_hgn.planning.strips import STRIPSProblem

@pytest.mark.parametrize(
    "problem", [

    ]
)
def test_delete_relaxation_hypergraph_view(problem: STRIPSProblem):
    hypergraph_view = DeleteRelaxationHypergraphView(problem)

    # A single node corresponds to a single proposition
    assert hypergraph_view.nodes == problem.propositions

    for idx, hyperedge in enumerate(hypergraph_view.hyperedges):
        # Get action from STRIPS problem
        action = problem.actions[idx]

        # Check properties are maintained
        assert hyperedge.name == action.name
        assert hyperedge.weight == action.cost
        assert hyperedge.senders == action.preconditions
        assert hyperedge.receivers == action.add_effects
