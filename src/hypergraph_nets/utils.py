from typing import List

import torch

from hypergraph_nets.hypergraphs import HypergraphsTuple


def concat(input_hypergraphs: List[HypergraphsTuple], axis):
    """
    Concatenates hypergraph tuples along a given axis.

    Based heavily off https://github.com/deepmind/graph_nets/blob/master/graph_nets/utils_tf.py#L331

    :param input_hypergraphs:
    :param axis:
    :return: concated HypergraphsTuple
    """
    if axis != 1:
        raise NotImplementedError("Only axis=1 supported at the moment")

    if len(input_hypergraphs) == 1:
        return input_hypergraphs[0]

    # Pre-concatenation
    nodes = [
        h_tup.nodes for h_tup in input_hypergraphs if h_tup.nodes is not None
    ]
    edges = [
        h_tup.edges for h_tup in input_hypergraphs if h_tup.edges is not None
    ]
    globals_ = [
        h_tup.globals
        for h_tup in input_hypergraphs
        if h_tup.globals is not None
    ]

    # Concatenated results
    nodes = torch.cat(nodes, dim=axis) if nodes else None
    edges = torch.cat(edges, dim=axis) if edges else None
    globals_ = torch.cat(globals_, dim=axis) if globals_ else None

    output = input_hypergraphs[0].replace(
        nodes=nodes, edges=edges, globals=globals_
    )
    if axis != 0:
        return output
