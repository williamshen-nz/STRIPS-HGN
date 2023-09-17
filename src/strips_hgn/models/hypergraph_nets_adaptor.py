import logging
from itertools import cycle, islice
from typing import Callable, List, Optional

import torch

from hypergraph_nets.hypergraphs import (
    EDGES,
    GLOBALS,
    N_EDGE,
    N_NODE,
    NODES,
    RECEIVERS,
    SENDERS,
    ZERO_PADDING,
    HypergraphsTuple,
)
from strips_hgn.hypergraph.hypergraph_view import HypergraphView

_log = logging.getLogger(__name__)


def _validate_features(features, expected_size, label):
    """
    Check features conform to expected size. Only support lists for now,
    no np.ndarray or torch.Tensor
    """
    if features is None:
        return

    if isinstance(features, torch.Tensor):
        assert features.shape[0] == expected_size
    else:
        raise NotImplementedError(
            f"Unexpected features type of {type(features)} for {label}"
        )


def repeat_up_to_k(lst, k):
    """
    Repeats a list so that it is of length k:
    https://stackoverflow.com/a/39863275

    e.g. _repeat_up_to_k([1,2,3], 10)
         => [1,2,3,1,2,3,1,2,3,1]
    """
    assert k >= len(lst)
    return list(islice(cycle(lst), k))


def pad_with_obj_up_to_k(lst, k, pad_with=-1):
    """
    Pads a list with an object so resulting length is k
    e.g. _pad_with_zeros_up_to_k([1,2,3], 5, 0)
         => [1,2,3,0,0]
    """
    assert k >= len(lst)
    return lst + (k - len(lst)) * [pad_with]


# noinspection PyArgumentList
def hypergraph_view_to_hypergraphs_tuple(
    hypergraph: HypergraphView,
    receiver_k: int,
    sender_k: int,
    node_features: Optional[torch.Tensor] = None,
    edge_features: Optional[torch.Tensor] = None,
    global_features: Optional[torch.Tensor] = None,
    pad_func: Callable[[list, int], list] = pad_with_obj_up_to_k,
) -> HypergraphsTuple:
    """
    Convert a Delete-Relaxation Task to a Hypergraphs Tuple (with
    node/edge/global features)

    :param hypergraph: HypergraphView
    :param receiver_k: maximum number of receivers for a hyperedge, receivers will be repeated to fit k
    :param sender_k: maximum number of senders for a hyperedge, senders will be repeated to fit k
    :param node_features: node features as a torch.Tensor
    :param edge_features: edge features as a torch.Tensor
    :param global_features: global features as a torch.Tensor
    :param pad_func: function for handling different number of sender/receiver nodes
    :return: parsed HypergraphsTuple
    """
    # Receivers are the additive effects for each action
    receivers = torch.LongTensor(
        [
            pad_func(
                [
                    # FIXME
                    hypergraph.node_to_idx(atom)
                    for atom in sorted(hyperedge.receivers)
                ],
                receiver_k,
            )
            for hyperedge in hypergraph.hyperedges
        ]
    )

    # Senders are preconditions for each action
    senders = torch.LongTensor(
        [
            pad_func(
                [
                    # FIXME
                    hypergraph.node_to_idx(atom)
                    for atom in sorted(hyperedge.senders)
                ],
                sender_k,
            )
            for hyperedge in hypergraph.hyperedges
        ]
    )

    # Validate features
    _validate_features(node_features, len(hypergraph.nodes), "Nodes")
    _validate_features(edge_features, len(hypergraph.hyperedges), "Edges")
    if global_features is not None:
        _validate_features(global_features, len(global_features), "Global")

    params = {
        N_NODE: torch.LongTensor([len(hypergraph.nodes)]),
        N_EDGE: torch.LongTensor([len(hypergraph.hyperedges)]),
        # Hyperedge connection information
        RECEIVERS: receivers,
        SENDERS: senders,
        # Features, set to None
        NODES: node_features,
        EDGES: edge_features,
        GLOBALS: global_features,
        ZERO_PADDING: pad_func == pad_with_obj_up_to_k,
    }

    return HypergraphsTuple(**params)


def merge_hypergraphs_tuple(
    graphs_tuple_list: List[HypergraphsTuple]
) -> HypergraphsTuple:
    """
    Merge multiple HypergraphsTuple (each representing one hypergraph)
    together into one - i.e. batch them up
    """
    assert len(graphs_tuple_list) > 0

    def _stack_features(attr_name, force_matrix=True):
        """ Stack matrices on top of each other """
        features = [
            getattr(h_tup, attr_name)
            for h_tup in graphs_tuple_list
            if getattr(h_tup, attr_name) is not None
        ]
        if len(features) == 0:
            return None
        else:
            stacked = torch.cat(features)
            if force_matrix and len(stacked.shape) == 1:
                stacked = stacked.reshape(-1, 1)
            return stacked

    # New tuple attributes
    n_node, n_edge, receivers, senders, nodes, edges, globals_ = (
        _stack_features(attr_name, force_matrix)
        for attr_name, force_matrix in [
            (N_NODE, False),
            (N_EDGE, False),
            (RECEIVERS, True),
            (SENDERS, True),
            (NODES, True),
            (EDGES, True),
            (GLOBALS, True),
        ]
    )

    # Need to increase indices for each hypergraph based on number of nodes
    # in previous hypergraphs
    n_edge_cumsum = torch.cumsum(n_edge, dim=0)
    n_node_cumsum = torch.cumsum(n_node, dim=0)
    for idx, (n_edge_prev, n_edge_cur) in enumerate(
        zip(n_edge_cumsum, n_edge_cumsum[1:])
    ):
        receivers[n_edge_prev:n_edge_cur][
            receivers[n_edge_prev:n_edge_cur] != -1
        ] += n_node_cumsum[idx]
        senders[n_edge_prev:n_edge_cur][
            senders[n_edge_prev:n_edge_cur] != -1
        ] += n_node_cumsum[idx]

    # Check padding consistent across hypergraphs
    assert len(set(h.zero_padding for h in graphs_tuple_list)) == 1
    zero_padding = graphs_tuple_list[0].zero_padding

    # Check general sizes have been maintained
    assert len(n_node) == len(n_edge) == len(graphs_tuple_list)
    assert receivers.shape[0] == senders.shape[0] == torch.sum(n_edge)

    if edges is not None:
        assert edges.shape[0] == torch.sum(n_edge)
    if nodes is not None:
        assert nodes.shape[0] == torch.sum(n_node)
    if globals_ is not None:
        assert globals_.shape[0] == len(graphs_tuple_list)

    return HypergraphsTuple(
        **{
            N_NODE: n_node,
            N_EDGE: n_edge,
            # Hyperedge connection information
            RECEIVERS: receivers,
            SENDERS: senders,
            # Features, turn them to tensors
            NODES: nodes,
            EDGES: edges,
            GLOBALS: globals_,
            ZERO_PADDING: zero_padding,
        }
    )
