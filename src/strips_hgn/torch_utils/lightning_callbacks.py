import logging
import time

from pytorch_lightning.callbacks import EarlyStopping

_log = logging.getLogger(__name__)


class EarlyStoppingWithMaxTrainingTime(EarlyStopping):
    """
    Early Stopping with Max Training Time
    """

    def __init__(self, patience: int, max_training_time: int):
        """
        Parameters
        ----------
        patience: number of epochs with no improvement after which training
            will be stopped
        max_training_time: maximum training time in seconds, elapsed time will
            be checked at the end of every epoch
        """
        super().__init__(patience=patience)

        assert max_training_time > 0
        self._max_training_time = max_training_time
        self._num_epochs_processed = 0
        self._start_time = None

        _log.info(
            f"Early stopping patience = {patience} epochs, "
            f"max training time set to {max_training_time}s"
        )

    def on_train_start(self, trainer, pl_module):
        super().on_train_start(trainer, pl_module)
        # Start timer
        self._start_time = time.perf_counter()

    def on_epoch_end(self, trainer, pl_module):
        # Lightning may callback here during an epoch as well (it's a bug),
        # so use the pl_module to determine the current epoch
        self._num_epochs_processed = pl_module.current_epoch + 1

        # Check if early stopping condition satisfied
        early_stop = super().on_epoch_end(trainer, pl_module)
        if early_stop:
            _log.info(
                f"Early stop activated, patience of {self.patience} "
                "epochs exceeded"
            )
            return early_stop

        # Check if we have exceeded max training time, taking into account
        # the average time required for an epoch so we try not to exceed it
        time_elapsed = time.perf_counter() - self._start_time
        avg_time_per_epoch = time_elapsed / self._num_epochs_processed

        if time_elapsed + avg_time_per_epoch >= self._max_training_time:
            _log.info(
                f"Stopping training - elapsed time = {time_elapsed}s, "
                f"avg time per epoch = {avg_time_per_epoch}s"
            )
            return True

        return False
