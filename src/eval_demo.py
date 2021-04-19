import json
from collections import namedtuple
from typing import FrozenSet, Type

import torch

from hypergraph_nets.hypergraphs import HypergraphsTuple
from strips_hgn.features import HyperedgeFeatureMapper, NodeFeatureMapper
from strips_hgn.features.node_features import PropositionInStateAndGoal
from strips_hgn.hypergraph.delete_relaxation import (
    DeleteRelaxationHypergraphView,
)
from strips_hgn.hypergraph.hypergraph_view import HypergraphView
from strips_hgn.models.heuristic import STRIPSHGNHeuristic
from strips_hgn.models.hypergraph_nets_adaptor import (
    hypergraph_view_to_hypergraphs_tuple,
)
from strips_hgn.models.strips_hgn import STRIPSHGN
from strips_hgn.planning import (
    get_strips_problem,
    STRIPSProblem,
    PlannerForEvaluation,
    Proposition,
)


def _get_node_feature_mapper(
    node_feature_mapper_cls: Type[NodeFeatureMapper],
    current_state: FrozenSet[Proposition],
    problem: STRIPSProblem,
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
    if node_feature_mapper_cls == PropositionInStateAndGoal:
        # Create node feature mapper for current state and the goal
        return PropositionInStateAndGoal(
            current_state=current_state, goal_state=problem.goals
        )
    else:
        raise RuntimeError(
            f"Unsupported node feature mapper {node_feature_mapper_cls}"
        )


def _get_input_hypergraphs_tuple(
    current_state: FrozenSet[Proposition],
    hypergraph: HypergraphView,
    max_receivers: int,
    max_senders: int,
    node_feature_mapper_cls: Type[NodeFeatureMapper],
    hyperedge_feature_mapper_cls: Type[HyperedgeFeatureMapper],
) -> HypergraphsTuple:
    """
    Get the input HypergraphsTuple to the STRIPS-HGN for the current state
    """
    # Instantiate node feature mapper and get node features
    node_feature_mapper = _get_node_feature_mapper(
        node_feature_mapper_cls, current_state, hypergraph.problem
    )
    node_features = torch.tensor(
        hypergraph.node_features(node_feature_mapper), dtype=torch.float32
    )

    # Instantiate hyperedge feature mapper and get hyperedge features
    # Hyperedge feature mappers do not require context to the current state
    hyperedge_feature_mapper = hyperedge_feature_mapper_cls()
    hyperedge_features = torch.tensor(
        hypergraph.hyperedge_features(hyperedge_feature_mapper),
        dtype=torch.float32,
    )

    # Get the HypergraphsTuple
    return hypergraph_view_to_hypergraphs_tuple(
        hypergraph=hypergraph,
        receiver_k=max_receivers,
        sender_k=max_senders,
        node_features=node_features,
        edge_features=hyperedge_features,
    )


def eval_demo(
    domain_pddl: str, problem_pddl: str, checkpoint: str, num_steps: int = 10
):
    """ Runs STRIPS-HGN for initial state of problem """
    # Generate the STRIPSProblem and get the DeleteRelaxationHypergraphView
    problem: STRIPSProblem = get_strips_problem(domain_pddl, problem_pddl)
    hypergraph = DeleteRelaxationHypergraphView(problem)

    # Load STRIPS-HGN model and setup evaluation mode
    model: STRIPSHGN = STRIPSHGN.load_from_checkpoint(checkpoint)
    model.setup_prediction_mode()

    hparams = model.hparams
    print(
        "STRIPS-HGN hparams",
        json.dumps(hparams.__dict__, indent=2, default=str),
    )

    def state_to_input_h_tup(state):
        # Function that maps from a state to a HypergraphsTuple
        return _get_input_hypergraphs_tuple(
            current_state=state,
            hypergraph=hypergraph,
            max_receivers=hparams.receiver_k,
            max_senders=hparams.sender_k,
            node_feature_mapper_cls=hparams.node_feature_mapper_cls,
            hyperedge_feature_mapper_cls=hparams.hyperedge_feature_mapper_cls,
        )

    # There are two ways we can call the heuristic
    # 1. Call the STRIPSHGN directly with a HypergraphsTuple
    # 2. Use the STRIPSHGNHeuristic wrapper

    # Option 1: Call the STRIPSHGN directly with a HypergraphsTuple
    input_h_tuple = state_to_input_h_tup(problem.initial_state)
    output_h_tuple = model(input_h_tuple, num_steps)
    assert len(output_h_tuple) == 1

    heuristic_val = output_h_tuple[0].globals.item()
    print(f"STRIPSHGN h(s_0) = {heuristic_val}")

    # Option 2: Use the STRIPSHGNHeuristic wrapper
    # Create a STRIPS-HGN heuristic
    strips_hgn_heuristic = STRIPSHGNHeuristic(
        model=model,
        state_to_input_hypergraphs_tuple=state_to_input_h_tup,
        planner=PlannerForEvaluation.pyperplan,
        num_steps=num_steps,
    )

    # Wrap the initial state in a Node object because that's how pyperplan does
    # things
    Node = namedtuple("Node", "state")
    initial_state_node = Node(problem.initial_state)
    heuristic_val = strips_hgn_heuristic(initial_state_node)
    print(f"STRIPSHGNHeuristic h(s_0) = {heuristic_val}")


if __name__ == "__main__":
    eval_demo(
        domain_pddl="../benchmarks/blocks-slaney/domain.pddl",
        problem_pddl="../benchmarks/blocks-slaney/blocks10/task01.pddl",
        checkpoint="../results/blocksworld-example-new.ckpt",
    )
