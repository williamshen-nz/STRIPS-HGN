from torch import nn

from hypergraph_nets.blocks import EdgeBlock, GlobalBlock, NodeBlock
from hypergraph_nets.hypergraphs import HypergraphsTuple


"""
General modules using the edge, node and global blocks defined in blocks.py

Based heavily off models implemented in Tensorflow by Deepmind.
https://github.com/deepmind/graph_nets/blob/master/graph_nets/modules.py

Each PyTorch module here takes a HypergraphsTuple as input, and returns a
HypergraphsTuple as output (with the updated values).

@author William Shen
"""


class GraphNetwork(nn.Module):
    """
    General graph network. First apply edge block, then node block, and finally, the global block.
    """

    def __init__(
        self,
        edge_model,
        node_model,
        global_model,
        global_blocks_use_globals: bool = False,
    ):
        super(GraphNetwork, self).__init__()
        self._edge_block = EdgeBlock(edge_model, use_globals=False)
        self._node_block = NodeBlock(node_model, use_globals=False)
        self._global_block = GlobalBlock(
            global_model, use_globals=global_blocks_use_globals
        )

    def forward(self, hypergraph: HypergraphsTuple):
        h_edge = self._edge_block(hypergraph)
        h_node = self._node_block(h_edge)
        h_global = self._global_block(h_node)
        return h_global


class GraphIndependent(nn.Module):
    """
    Each model is applied to the graph elements independently.
    """

    def __init__(self, edge_model, node_model, global_model):
        super(GraphIndependent, self).__init__()
        self._edge_model = edge_model
        self._node_model = node_model
        self._global_model = global_model

    def forward(self, hypergraph: HypergraphsTuple):
        return hypergraph.replace(
            edges=self._edge_model(hypergraph.edges),
            nodes=self._node_model(hypergraph.nodes),
            globals=self._global_model(hypergraph.globals),
        )
