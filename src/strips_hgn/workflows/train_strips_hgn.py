import logging
import os
from typing import Optional

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from strips_hgn.models.strips_hgn import STRIPSHGN
from strips_hgn.torch_utils.lightning_callbacks import (
    EarlyStoppingWithMaxTrainingTime,
)
from strips_hgn.utils.timer import TimedOperation

_log = logging.getLogger(__name__)


class TrainSTRIPSHGNWorkflow(object):

    _NUM_TOP_MODELS_TO_SAVE = 10
    _CHECKPOINT_FORMAT = "{epoch}-{val_loss:.4f}"

    def __init__(
        self,
        strips_hgn: STRIPSHGN,
        max_training_time: int,
        max_num_epochs: int,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        experiments_dir: str,
        prefix: str,
        early_stopping_patience: int = 10,
        fast_dev_run: bool = False,
    ):
        # Set model to training mode
        strips_hgn.train()
        self.model = strips_hgn

        self.max_training_time = max_training_time
        self.max_epochs = max_num_epochs
        self.prefix = prefix
        self.checkpoint_dir = os.path.join(experiments_dir, prefix, "")

        # Dataloaders
        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader

        # Lightning stuff (callback, logger, etc.)
        self._early_stop_callback = EarlyStoppingWithMaxTrainingTime(
            patience=early_stopping_patience,
            max_training_time=max_training_time,
        )
        self._model_checkpoint_callback = ModelCheckpoint(
            filepath=os.path.join(
                self.checkpoint_dir, TrainSTRIPSHGNWorkflow._CHECKPOINT_FORMAT
            ),
            monitor="val_loss",
            mode="min",
            save_top_k=TrainSTRIPSHGNWorkflow._NUM_TOP_MODELS_TO_SAVE,
            verbose=True,
        )
        self._logger = TensorBoardLogger(
            save_dir=experiments_dir, name="", version=prefix
        )

        # Lightning trainer
        self.trainer = Trainer(
            early_stop_callback=self._early_stop_callback,
            max_epochs=self.max_epochs,
            checkpoint_callback=self._model_checkpoint_callback,
            logger=self._logger,
            weights_summary="full",
            fast_dev_run=fast_dev_run,
        )

        # Whether network has already been trained
        self._workflow_ran = False

        # Log some debugging info
        # Number of learnable parameters
        torch_total_params = sum(p.numel() for p in self.model.parameters())
        _log.info(
            "STRIPS-HGN total number of learnable parameters: "
            f"{torch_total_params}"
        )

        # Size of training and validation data loader
        _log.info(
            f"Training dataloader size: {len(self._train_dataloader)}, "
            f"Validation dataloader size: {len(self._val_dataloader)}"
        )

        # Training info
        _log.info(
            f"Max training {self.max_training_time} seconds or "
            f"{self.max_epochs} epochs, whichever is reached first"
        )

    @property
    def best_val_loss(self):
        assert self._workflow_ran
        return min(self._model_checkpoint_callback.best_k_models.values())

    @property
    def best_val_loss_checkpoint(self) -> Optional[str]:
        assert self._workflow_ran

        # Get the filepath with the lowest validation loss
        filepath = [
            filepath
            for filepath, val_loss in self._model_checkpoint_callback.best_k_models.items()
            if val_loss == self.best_val_loss
        ]
        if len(filepath) > 1:
            raise RuntimeError("Found 2 checkpoints for best val loss?")

        return filepath[0] if filepath else None

    @property
    def current_epoch(self) -> int:
        return self.trainer.current_epoch

    def run(self):
        if self._workflow_ran:
            raise RuntimeError("Cannot run training workflow more than once")

        with TimedOperation("TrainSTRIPSHGNWorkflow.TotalTime"):
            self.trainer.fit(
                model=self.model,
                train_dataloader=self._train_dataloader,
                val_dataloaders=self._val_dataloader,
            )

            self._workflow_ran = True
