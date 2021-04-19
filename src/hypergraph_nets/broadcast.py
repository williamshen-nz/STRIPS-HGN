import torch

from hypergraph_nets.hypergraphs import HypergraphsTuple


"""
Broadcast operations for propagating features from edges/nodes/globals to
relevant elements of the next edges/nodes/globals. Modified for PyTorch

Based heavily off models implemented in Tensorflow by Deepmind.
https://github.com/deepmind/graph_nets/blob/master/graph_nets/blocks.py

@author William Shen
"""


def broadcast_receiver_nodes_to_edges(
    hypergraph: HypergraphsTuple
) -> torch.Tensor:
    """ Sort of equivalent to tf.gather(hypergraph.nodes, hypergraph.receivers) """
    if hypergraph.zero_padding:
        return hypergraph.nodes_with_zero_last()[hypergraph.receivers].reshape(
            hypergraph.total_n_edge, -1
        )
    else:
        return hypergraph.nodes[hypergraph.receivers].reshape(
            hypergraph.total_n_edge, -1
        )


def broadcast_sender_nodes_to_edges(
    hypergraph: HypergraphsTuple
) -> torch.Tensor:
    """ Sort of equivalent to tf.gather(hypergraph.nodes, hypergraph.senders) """
    if hypergraph.zero_padding:
        return hypergraph.nodes_with_zero_last()[hypergraph.senders].reshape(
            hypergraph.total_n_edge, -1
        )
    else:
        return hypergraph.nodes[hypergraph.senders].reshape(
            hypergraph.total_n_edge, -1
        )


def broadcast_globals_to_edges(hypergraph: HypergraphsTuple) -> torch.Tensor:
    """ Broadcast global features to edges of a hypergraph """
    return hypergraph.globals.repeat_interleave(hypergraph.n_edge, dim=0)


def broadcast_globals_to_nodes(hypergraph: HypergraphsTuple) -> torch.Tensor:
    """ Broadcast global features to nodes of a hypergraph """
    return hypergraph.globals.repeat_interleave(hypergraph.n_node, dim=0)
