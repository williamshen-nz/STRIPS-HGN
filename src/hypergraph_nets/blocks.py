import torch
from torch import nn

from hypergraph_nets.aggregators import (
    EdgesToGlobalsAggregator,
    NodesToGlobalsAggregator,
    ReceivedEdgesToNodesAggregator,
    SentEdgesToNodesAggregator,
)
from hypergraph_nets.broadcast import (
    broadcast_globals_to_edges,
    broadcast_globals_to_nodes,
    broadcast_receiver_nodes_to_edges,
    broadcast_sender_nodes_to_edges,
)
from hypergraph_nets.hypergraphs import HypergraphsTuple
from hypergraph_nets.reducers import torch_unsorted_segment_sum


"""
Edge, Node and Global Blocks for Graph Networks in PyTorch

Based heavily off models implemented in Tensorflow by Deepmind.
https://github.com/deepmind/graph_nets/blob/master/graph_nets/blocks.py

@author William Shen
"""


class EdgeBlock(nn.Module):
    """
    Block that updates features of edge based on at least one of:
    - Previous edge features
    - Features of adjacent nodes (receiver and/or sender nodes)
    - Global features of graph

    For details, see: https://github.com/deepmind/graph_nets/blob/master/graph_nets/blocks.py#L373
    """

    def __init__(
        self,
        edge_model,
        use_edges=True,
        use_receiver_nodes=True,
        use_sender_nodes=True,
        use_globals=True,
    ):
        super(EdgeBlock, self).__init__()
        self._edge_model = edge_model
        # Options
        assert (
            use_edges or use_receiver_nodes or use_sender_nodes or use_globals
        )
        self._use_edges = use_edges
        self._use_receiver_nodes = use_receiver_nodes
        self._use_sender_nodes = use_sender_nodes
        self._use_globals = use_globals

    def forward(self, hypergraph: HypergraphsTuple):
        edges_to_collect = []

        if self._use_edges:
            edges_to_collect.append(hypergraph.edges)

        if self._use_receiver_nodes:
            edges_to_collect.append(
                broadcast_receiver_nodes_to_edges(hypergraph)
            )

        if self._use_sender_nodes:
            edges_to_collect.append(
                broadcast_sender_nodes_to_edges(hypergraph)
            )

        if self._use_globals:
            edges_to_collect.append(broadcast_globals_to_edges(hypergraph))

        # Concatenate so we maintain number of edges, just extend
        # dimensionality of each edge
        collected_edges = torch.cat(edges_to_collect, -1)
        updated_edges = self._edge_model(collected_edges)
        return hypergraph.replace(edges=updated_edges)


class NodeBlock(nn.Module):
    """
    Block that updates features of node based on at least one of:
    - Previous node features
    - Aggregated features of adjacent edges
    - Global features of graph

    For details, see https://github.com/deepmind/graph_nets/blob/master/graph_nets/blocks.py#L458
    """

    def __init__(
        self,
        node_model,
        use_received_edges=True,
        use_sent_edges=False,
        use_nodes=True,
        use_globals=True,
        received_edges_reducer=torch_unsorted_segment_sum,
        sent_edges_reducer=torch_unsorted_segment_sum,
    ):
        super(NodeBlock, self).__init__()
        self._node_model = node_model
        # Options
        assert use_received_edges or use_sent_edges or use_nodes or use_globals
        self._use_received_edges = use_received_edges
        self._use_sent_edges = use_sent_edges
        self._use_nodes = use_nodes
        self._use_globals = use_globals

        if self._use_received_edges:
            assert received_edges_reducer
            self._received_edges_aggregator = ReceivedEdgesToNodesAggregator(
                received_edges_reducer
            )
        if self._use_sent_edges:
            assert sent_edges_reducer
            self._sent_edges_aggregator = SentEdgesToNodesAggregator(
                sent_edges_reducer
            )

    def forward(self, hypergraph: HypergraphsTuple):
        nodes_to_collect = []

        if self._use_received_edges:
            nodes_to_collect.append(
                self._received_edges_aggregator(hypergraph)
            )

        if self._use_sent_edges:
            nodes_to_collect.append(self._sent_edges_aggregator(hypergraph))

        if self._use_nodes:
            nodes_to_collect.append(hypergraph.nodes)

        if self._use_globals:
            nodes_to_collect.append(broadcast_globals_to_nodes(hypergraph))

        # Concatenate so we maintain number of node, just extend
        # dimensionality of each node
        collected_nodes = torch.cat(nodes_to_collect, -1)
        updated_nodes = self._node_model(collected_nodes)
        return hypergraph.replace(nodes=updated_nodes)


class GlobalBlock(nn.Module):
    """
    Block that updates global features of a graph based on at least one of:
      - Previous global features
      - Aggregated features of edges of graph
      - Aggregated features of nodes of graph

    For details, see https://github.com/deepmind/graph_nets/blob/master/graph_nets/blocks.py#L568
    """

    def __init__(
        self,
        global_model,
        use_edges=True,
        use_nodes=True,
        use_globals=True,
        nodes_reducer=torch_unsorted_segment_sum,
        edges_reducer=torch_unsorted_segment_sum,
    ):
        super(GlobalBlock, self).__init__()
        self._global_model = global_model

        # Options
        assert use_edges or use_nodes or use_globals
        self._use_edges = use_edges
        self._use_nodes = use_nodes
        self._use_globals = use_globals

        if self._use_edges:
            self._edges_aggregator = EdgesToGlobalsAggregator(edges_reducer)
        if self._use_nodes:
            self._nodes_aggregator = NodesToGlobalsAggregator(nodes_reducer)

    def forward(self, hypergraph: HypergraphsTuple):
        globals_to_collect = []

        if self._use_edges:
            globals_to_collect.append(self._edges_aggregator(hypergraph))

        if self._use_nodes:
            globals_to_collect.append(self._nodes_aggregator(hypergraph))

        if self._use_globals:
            globals_to_collect.append(hypergraph.globals)

        # Concatenate so we maintain number of globals, just extend
        # dimensionality of each global
        collected_globals = torch.cat(globals_to_collect, -1)
        updated_globals = self._global_model(collected_globals)
        return hypergraph.replace(globals=updated_globals)
