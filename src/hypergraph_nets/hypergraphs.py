from collections import namedtuple

import torch

# from torch_config import get_device

"""
Defines hypergraph-structured data. In comparison to graph_nets GraphsTuple, we extend
it to Hypergraphs. Mainly shown in other method implementations (e.g. broadcasting)

Based heavily off models implemented in Tensorflow by Deepmind.
https://github.com/deepmind/graph_nets/blob/master/graph_nets/graphs.py

@author William Shen

The following text below is directly copied from the link listed above for clarity's sake.
Modifications have been made to tailor the description to *hypergraphs*.

===
The main purpose of the `HypergraphsTuple` is to represent multiple hypergraphs with
different shapes and sizes in a way that supports batched processing.

This module first defines the string constants which are used to represent
graph(s) as tuples or dictionaries: `N_NODE, N_EDGE, NODES, EDGES, RECEIVERS,
SENDERS, GLOBALS`.

This representation could typically take the following form, for a batch of
`n_graphs` hypergraphs stored in a `HypergraphsTuple` called hypergraph:
  - N_NODE: The number of nodes per graph. It is a vector of integers with shape
    `[n_graphs]`, such that `graph.N_NODE[i]` is the number of nodes in the i-th
    graph.
  - N_EDGE: The number of edges per graph. It is a vector of integers with shape
    `[n_graphs]`, such that `graph.N_NODE[i]` is the number of edges in the i-th
    graph.
  - NODES: The nodes features. It is either `None` (the graph has no node
    features), or a vector of shape `[n_nodes] + node_shape`, where
    `n_nodes = sum(graph.N_NODE)` is the total number of nodes in the batch of
    graphs, and `node_shape` represents the shape of the features of each node.
    The relative index of a node from the batched version can be recovered from
    the `graph.N_NODE` property. For instance, the second node of the third
    graph will have its features in the
    `1 + graph.N_NODE[0] + graph.N_NODE[1]`-th slot of graph.NODES.
    Observe that having a `None` value for this field does not mean that the
    graphs have no nodes, only that they do not have features.
  - EDGES: The edges features. It is either `None` (the graph has no edge
    features), or a vector of shape `[n_edges] + edge_shape`, where
    `n_edges = sum(graph.N_EDGE)` is the total number of edges in the batch of
    graphs, and `edge_shape` represents the shape of the features of each edge.
    The relative index of an edge from the batched version can be recovered from
    the `graph.N_EDGE` property. For instance, the third edge of the third
    graph will have its features in the `2 + graph.N_EDGE[0] + graph.N_EDGE[1]`-
    th slot of graph.EDGES.
    Observe that having a `None` value for this field does not necessarily mean
    that the graph has no edges, only that they do not have features.
  - RECEIVERS: The indices of the receiver nodes, for each edge. It is either
    `None` (if the graph has no edges), or a matrix of integers of shape
    `[n_edges]`, such that `graph.RECEIVERS[i]` are the indices of the nodes
    receiving from the i-th edge.
    Observe that the index is absolute (in other words, cumulative), i.e.
    `graphs.RECEIVERS` take value in `[0, n_nodes]`. For instance, an edge
    connecting the vertices with relative indices 2 and 3 in the second graph of
    the batch would have a `RECEIVERS` value of `3 + graph.N_NODE[0]`.
    If `graphs.RECEIVERS` is `None`, then `graphs.EDGES` and `graphs.SENDERS`
    should also be `None`.
  - SENDERS: The indices of the sender nodes, for each edge. It is either
    `None` (if the graph has no edges), or a matrix of integers of shape
    `[n_edges]`, such that `graph.SENDERS[i]` are the indices of the nodes
    sending from the i-th edge.
    Observe that the index is absolute, i.e. `graphs.RECEIVERS` take value in
    `[0, n_nodes]`. For instance, an edge connecting the vertices with relative
    indices 1 and 3 in the third graph of the batch would have a `SENDERS` value
    of `1 + graph.N_NODE[0] + graph.N_NODE[1]`.
    If `graphs.SENDERS` is `None`, then `graphs.EDGES` and `graphs.RECEIVERS`
    should also be `None`.
  - GLOBALS: The global features of the graph. It is either `None` (the graph
    has no global features), or a vector of shape `[n_graphs] + global_shape`
    representing graph level features.

"""

# Note, the only shape that has changed is RECEIVERS and SENDERS for the hyperedges
# Shape: [n_nodes] + node_shape
NODES = "nodes"

# Shape: [n_edges] + edge_shape
EDGES = "edges"

# Shape: [n_edges] + (number of receiving neighbours in specific edge)
# Type: List[Tensor]
RECEIVERS = "receivers"

# Shape: [n_edges] + (number of sending neighbours in specific edge)
# Type: List[Tensor]
SENDERS = "senders"

# Shape: [n_graphs] + global_shape
GLOBALS = "globals"

# Shape: [n_graphs]
N_NODE = "n_node"
N_EDGE = "n_edge"

# bool
ZERO_PADDING = "zero_padding"

GRAPH_FEATURE_FIELDS = (NODES, EDGES, GLOBALS)
GRAPH_INDEX_FIELDS = (RECEIVERS, SENDERS)
GRAPH_DATA_FIELDS = (NODES, EDGES, RECEIVERS, SENDERS, GLOBALS)
GRAPH_NUMBER_FIELDS = (N_NODE, N_EDGE)
ALL_FIELDS = (
    NODES,
    EDGES,
    RECEIVERS,
    SENDERS,
    GLOBALS,
    N_NODE,
    N_EDGE,
    ZERO_PADDING,
)

_CANNOT_REPLACE = {N_NODE, N_EDGE, RECEIVERS, SENDERS, ZERO_PADDING}


class HypergraphsTuple(
    namedtuple(
        "HypergraphsTuple",
        [
            "nodes",
            "edges",
            "receivers",
            "senders",
            "globals",
            "n_node",
            "n_edge",
            "zero_padding",
        ],
    )
):
    """
    Modified from https://github.com/deepmind/graph_nets/blob/master/graph_nets/graphs.py#L126
    TODO: should probably validate fields
    """

    def __init__(self, *args, **kwargs):
        del args, kwargs
        super(HypergraphsTuple, self).__init__()

    def replace(self, **kwargs):
        if any(attr in kwargs for attr in _CANNOT_REPLACE):
            raise ValueError(f"Cannot replace any of {_CANNOT_REPLACE}!")
        output = self._replace(**kwargs)
        return output

    def nodes_with_zero_last(self):
        """
        Returns nodes tensor with a zero feature vector appended. Used for
        zero padding broadcasting
        """
        zero_row = torch.zeros(1, *self.nodes.shape[1:])  # .to(get_device())
        return torch.cat((self.nodes, zero_row), 0)

    @property
    def total_n_edge(self):
        return torch.sum(self.n_edge)

    @property
    def total_n_node(self):
        return torch.sum(self.n_node)

    @property
    def num_hypergraphs(self):
        return self.n_node.shape[0]
