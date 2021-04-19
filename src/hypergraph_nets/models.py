from torch import nn

from hypergraph_nets.hypergraphs import HypergraphsTuple
from hypergraph_nets.modules import GraphIndependent, GraphNetwork
from hypergraph_nets.utils import concat


"""
General models using graph network modules that utilise MDPs as intermediate learnable functions

Based heavily off models implemented in Tensorflow by Deepmind.
https://github.com/deepmind/graph_nets/blob/master/graph_nets/demos/models.py

@author William Shen
"""


# Instead of Lambda's because pickle is stupid
def _identity_function(x):
    return x


def _none_function(_):
    return None


def make_mlp_model(input_dim, hidden_units=32, num_layers=2):
    """ Simple MLP with a LayerNorm """
    assert num_layers == 2, "Only num layers == 2 supported atm"
    return nn.Sequential(
        nn.Linear(input_dim, hidden_units),
        nn.LeakyReLU(inplace=True),
        nn.Linear(hidden_units, hidden_units),
        nn.LeakyReLU(inplace=True),
        # Normalize over last dimension
        # nn.LayerNorm(hidden_units),
    )


class MLPGraphIndependent(nn.Module):
    """ GraphIndependent using MLPs as the block models """

    def __init__(
        self,
        edge_input_size=None,
        node_input_size=None,
        global_input_size=None,
        hidden_units: int = 32,
    ):
        super(MLPGraphIndependent, self).__init__()
        self._gn = GraphIndependent(
            make_mlp_model(edge_input_size, hidden_units)
            if edge_input_size
            else _identity_function,
            make_mlp_model(node_input_size, hidden_units)
            if node_input_size
            else _identity_function,
            make_mlp_model(global_input_size, hidden_units)
            if global_input_size
            else _identity_function,
        )

    def forward(self, hypergraph: HypergraphsTuple) -> HypergraphsTuple:
        return self._gn(hypergraph)


class MLPGraphNetwork(nn.Module):
    """ GraphNetwork using MLPs as the block models"""

    def __init__(
        self,
        edge_input_size=None,
        node_input_size=None,
        global_input_size=None,
        hidden_units: int = 32,
        global_blocks_use_globals: bool = False,
    ):
        super(MLPGraphNetwork, self).__init__()
        self._gn = GraphNetwork(
            make_mlp_model(edge_input_size, hidden_units)
            if edge_input_size
            else _identity_function,
            make_mlp_model(node_input_size, hidden_units)
            if node_input_size
            else _identity_function,
            make_mlp_model(global_input_size, hidden_units)
            if global_input_size
            else _identity_function,
            global_blocks_use_globals=global_blocks_use_globals,
        )

    def forward(self, hypergraph: HypergraphsTuple) -> HypergraphsTuple:
        return self._gn(hypergraph)


class EncodeProcessDecode(nn.Module):
    """
    Encode-process-decode as described in the Graph Networks paper.
    Modified for PyTorch from: https://github.com/deepmind/graph_nets/blob/master/graph_nets/demos/models.py#L72

    - 'Encoder' independently encodes edge, node and global attributes
    - 'Core' does message-passing based on encoder's output and core's previous output
    - 'Decoder' independently decodes edge, node and global attributes
    """

    def __init__(
        self,
        receiver_k: int,
        sender_k: int,
        hidden_size: int,
        edge_input_size=None,
        edge_output_size=None,
        node_input_size=None,
        node_output_size=None,
        global_input_size=None,
        global_output_size=None,
    ):
        super(EncodeProcessDecode, self).__init__()
        self._encoder = MLPGraphIndependent(
            edge_input_size=edge_input_size,
            node_input_size=node_input_size,
            global_input_size=global_input_size,
            hidden_units=hidden_size,
        )

        # Core Network
        self._core = MLPGraphNetwork(
            # (latent + latent0) + (nodes + receiver_nodes_to_edges + sender_nodes_to_edges)
            (hidden_size + hidden_size) * (1 + receiver_k + sender_k),
            # (edges shape) + (latent) [edges shape has been squished to output hidden_size]
            hidden_size + (hidden_size + hidden_size),
            # (edges shape + nodes shape) + latent (if global input is specified)
            2 * hidden_size + (hidden_size + hidden_size)
            if global_input_size
            else 2 * hidden_size,
            global_blocks_use_globals=True if global_input_size else False,
            hidden_units=hidden_size,
        )

        # Only decode if an output is required
        self._decoder = MLPGraphIndependent(
            edge_input_size=hidden_size if edge_output_size else None,
            node_input_size=hidden_size if node_output_size else None,
            global_input_size=hidden_size if global_output_size else None,
            hidden_units=hidden_size,
        )

        # Transform outputs into appropriate shapes
        edge_model = (
            nn.Sequential(
                nn.Linear(hidden_size, edge_output_size), nn.ReLU(inplace=True)
            )
            if edge_output_size
            else _none_function
        )
        node_model = (
            nn.Sequential(
                nn.Linear(hidden_size, node_output_size), nn.ReLU(inplace=True)
            )
            if node_output_size
            else _none_function
        )
        global_model = (
            nn.Sequential(
                nn.Linear(hidden_size, global_output_size),
                nn.ReLU(inplace=True),
            )
            if global_output_size
            else _none_function
        )
        self._output_transform = GraphIndependent(
            edge_model, node_model, global_model
        )

    def forward(
        self, hypergraph: HypergraphsTuple, steps: int, pred_mode: bool = False
    ):
        latent: HypergraphsTuple = self._encoder(hypergraph)
        latent0: HypergraphsTuple = latent

        # Output for each time step (if in prediction mode, there will only
        # be one element for the last step)
        output_ops = []

        for idx in range(steps):
            core_input = concat([latent0, latent], axis=1)
            latent = self._core(core_input)
            if not pred_mode or idx == steps - 1:
                decoded_op = self._decoder(latent)
                output_ops.append(self._output_transform(decoded_op))

        return output_ops
