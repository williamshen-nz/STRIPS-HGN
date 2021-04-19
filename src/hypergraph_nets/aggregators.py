import torch
from torch import nn

from hypergraph_nets.hypergraphs import HypergraphsTuple


"""

Aggregating functions as defined in the Graph Networks paper.
Used to propagate and reduce features from edges/nodes/globals to the
relevant new edges/nodes/globals. Re-implemented for PyTorch

Based heavily off models implemented in Tensorflow by Deepmind.
https://github.com/deepmind/graph_nets/blob/master/graph_nets/blocks.py

@author William Shen
"""


class EdgesToGlobalsAggregator(nn.Module):
    """ Aggregates edges into globals """

    def __init__(self, reducer):
        super(EdgesToGlobalsAggregator, self).__init__()
        self._reducer = reducer

    def forward(self, hypergraph: HypergraphsTuple):
        graph_index = torch.arange(hypergraph.num_hypergraphs)
        indices = graph_index.repeat_interleave(hypergraph.n_edge, dim=0)
        return self._reducer(
            hypergraph.edges, indices, hypergraph.num_hypergraphs
        )


class NodesToGlobalsAggregator(nn.Module):
    """ Aggregates nodes into globals """

    def __init__(self, reducer):
        super(NodesToGlobalsAggregator, self).__init__()
        self._reducer = reducer

    def forward(self, hypergraph: HypergraphsTuple):
        graph_index = torch.arange(hypergraph.num_hypergraphs)
        indices = graph_index.repeat_interleave(hypergraph.n_node, dim=0)
        return self._reducer(
            hypergraph.nodes, indices, hypergraph.num_hypergraphs
        )


class _EdgesToNodesAggregator(nn.Module):
    """ Aggregates sent or received edges into corresponding nodes"""

    def __init__(self, reducer, use_sent_edges):
        super(_EdgesToNodesAggregator, self).__init__()
        self._reducer = reducer
        self._use_sent_edges = use_sent_edges

    def forward(self, hypergraph: HypergraphsTuple):
        num_nodes = torch.sum(hypergraph.n_node)
        indices = (
            hypergraph.senders
            if self._use_sent_edges
            else hypergraph.receivers
        )
        return self._reducer(hypergraph.edges, indices, num_nodes)


class SentEdgesToNodesAggregator(_EdgesToNodesAggregator):
    """ Aggregates sent edges into corresponding sender nodes """

    def __init__(self, reducer):
        super(SentEdgesToNodesAggregator, self).__init__(
            reducer, use_sent_edges=True
        )


class ReceivedEdgesToNodesAggregator(_EdgesToNodesAggregator):
    """ Aggregates received edges into corresponding receiver nodes """

    def __init__(self, reducer):
        super(ReceivedEdgesToNodesAggregator, self).__init__(
            reducer, use_sent_edges=False
        )
