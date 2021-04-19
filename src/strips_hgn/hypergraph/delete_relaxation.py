from typing import Dict, List

from strips_hgn.hypergraph import Hyperedge, Node
from strips_hgn.hypergraph.hypergraph_view import HypergraphView
from strips_hgn.planning import STRIPSProblem


class DeleteRelaxationHypergraphView(HypergraphView):
    """
    Delete-Relaxation Hypergraph view of a STRIPS problem where:
      - A node corresponds with a single proposition
      - A hyperedge corresponds with a relaxed action, connecting the
        preconditions to the additive effects
    """

    def __init__(self, problem: STRIPSProblem):
        super().__init__(problem)

        # Each node corresponds to a single proposition
        self._nodes = self.problem.propositions
        self._node_to_idx: Dict[Node, int] = {
            node: idx for idx, node in enumerate(self.nodes)
        }

        # Each hyperedge corresponds to a relaxed action where the senders
        # are the preconditions and the receivers are the additive effects.
        # Hence, the negative effects are ignored.
        self._hyperedges = [
            Hyperedge(
                name=action.name,
                weight=action.cost,
                senders=action.preconditions,
                receivers=action.add_effects,
                # Used to store context of delete-effects for feature mappers
                context={"delete_effects": action.del_effects},
            )
            for action in self.problem.actions
        ]
        self._hyperedge_to_idx: Dict[Hyperedge, int] = {
            hyperedge: idx for idx, hyperedge in enumerate(self._hyperedges)
        }

    @property
    def nodes(self) -> List[Node]:
        return self._nodes

    def node_to_idx(self, node: Node) -> int:
        return self._node_to_idx[node]

    @property
    def hyperedges(self) -> List[Hyperedge]:
        return self._hyperedges

    def hyperedge_to_idx(self, hyperedge: Hyperedge) -> int:
        return self._hyperedge_to_idx[hyperedge]
