import json
import logging
from typing import List

import torch
from pytorch_lightning import LightningModule
from torch.nn import MSELoss
from torch.optim import Adam

from hypergraph_nets.hypergraphs import HypergraphsTuple
from hypergraph_nets.models import EncodeProcessDecode
from strips_hgn.config import DEFAULT_HIDDEN_SIZE

_log = logging.getLogger(__name__)


def _get_debug_dict(hparams):
    return {
        "global_feature_mapper": hparams.global_feature_mapper_cls.name(),
        "hyperedge_feature_mapper": hparams.hyperedge_feature_mapper_cls.name(),
        "node_feature_mapper": hparams.node_feature_mapper_cls.name(),
        "receiver_k": hparams.receiver_k,
        "sender_k": hparams.sender_k,
        "hidden_size": hparams.hidden_size,
    }


class STRIPSHGN(LightningModule):
    """
    Based off:
    https://pytorch-lightning.readthedocs.io/en/latest/introduction_guide.html
    """

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self._prediction_mode = False
        self._criterion = MSELoss()

        # Override hidden size if not specified
        if "hidden_size" not in hparams:
            _log.warning(
                "hidden_size not specified in hparams! "
                f"Overriding to {DEFAULT_HIDDEN_SIZE}"
            )
            hparams.hidden_size = DEFAULT_HIDDEN_SIZE

        # Setup HGN
        self.hgn = EncodeProcessDecode(
            receiver_k=hparams.receiver_k,
            sender_k=hparams.sender_k,
            hidden_size=hparams.hidden_size,
            global_input_size=hparams.global_feature_mapper_cls.input_size(),
            edge_input_size=hparams.hyperedge_feature_mapper_cls.input_size(),
            node_input_size=hparams.node_feature_mapper_cls.input_size(),
            # We are always predicting a single heuristic value
            global_output_size=1,
        )

        # Log hyperparameters
        _log.info(
            "STRIPS-HGN hparams:\n"
            f"{json.dumps(_get_debug_dict(hparams), indent=2)}"
        )

    def _calc_loss(
        self,
        pred_graphs: List[HypergraphsTuple],
        target_graph: HypergraphsTuple,
    ):
        """
        By calculating the average loss over all the predicted graphs, we
        try to minimise number of message passing steps to get to the target
        graph.
        """
        return calc_avg_loss(self._criterion, pred_graphs, target_graph)

    def setup_prediction_mode(self):
        # Set flag for prediction mode
        self._prediction_mode = True

    def forward(self, hypergraph: HypergraphsTuple, num_steps: int = 10):
        """
        Run the forward pass through the network

        Parameters
        ----------
        hypergraph: HypergraphsTuple, one or more hypergraphs
        num_steps: number of message passing steps for the STRIPS-HGN

        Returns
        -------
        Output for each time step in t = 1, ..., num_steps
        """
        return self.hgn.forward(
            hypergraph=hypergraph,
            steps=num_steps,
            pred_mode=self._prediction_mode,
        )

    def configure_optimizers(self):
        # As defined in Section 6.1 of the STRIPS-HGN Paper
        return Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

    def training_step(self, batch, batch_idx):
        input_graph, target_graph = batch

        # Run model and calculate loss
        pred_graphs = self(input_graph)
        loss = self._calc_loss(pred_graphs, target_graph)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([step["loss"] for step in outputs]).mean()
        tensorboard_logs = {"train_loss": avg_loss}
        return {"avg_train_loss": avg_loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        input_graph, target_graph = batch

        # Run model and calculate loss
        pred_graphs = self(input_graph)
        return {"val_loss": self._calc_loss(pred_graphs, target_graph)}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([step["val_loss"] for step in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs}


def calc_avg_loss(
    criterion,
    pred_graphs: List[HypergraphsTuple],
    target_graph: HypergraphsTuple,
):
    """
    Calculates average loss for a criterion over multiple predictions
    """
    accum_loss = criterion(target_graph.globals, pred_graphs[0].globals)

    for pass_idx in range(1, len(pred_graphs)):
        accum_loss += criterion(
            target_graph.globals, pred_graphs[pass_idx].globals
        )

    return accum_loss / len(pred_graphs)
