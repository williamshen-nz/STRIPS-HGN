import logging
from functools import wraps
from time import perf_counter
from typing import Dict, Optional, Union

from strips_hgn.utils import Number
from strips_hgn.utils.metrics import TimeMetric, metrics_logger

_log = logging.getLogger(__name__)


class Timer(object):
    """ Timer that can be started and paused. """

    def __init__(self, name: str):
        self.name: str = name
        self.stopped: bool = False
        self._accumulated_time = 0.0
        self._last_start_time = None

    def start(self) -> "Timer":
        if self._last_start_time:
            raise RuntimeError(
                f"Cannot start timer {self.name} that is already started!"
            )
        elif self.stopped:
            raise RuntimeError(
                f"Cannot start timer {self.name} that has been stopped!"
            )
        else:
            self._last_start_time = perf_counter()
            return self

    def pause(self):
        if self._last_start_time:
            self._accumulated_time += perf_counter() - self._last_start_time
            self._last_start_time = None
        else:
            raise RuntimeError("Timer has not been started!")

    def stop(self):
        self.pause()
        self.stopped = True

    @property
    def total_time(self):
        return self._accumulated_time


class TimedOperation(Timer):
    def __init__(
        self,
        name: str,
        context: Optional[Dict[str, Union[str, Number]]] = None,
        log_level=logging.INFO,
    ):
        super(TimedOperation, self).__init__(name)
        self.context = context if context else {}
        self._log_level = log_level

    def __call__(self, func):
        """ Allow timer to be used as a decorator """

        @wraps(func)
        def wrapped_timer(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return wrapped_timer

    def add_context(self, name: str, value: Union[str, Number]):
        if name in self.context:
            _log.warning(
                f"{name} already exists in TimedOperation "
                f"{self.name} context. Overriding."
            )
        self.context[name] = value

    def stop(self):
        """ Stop the timer and add it to the metrics logger"""
        super().stop()

        log_str = (
            f"Timer {self.name} stopped. Accumulated time: "
            f"{round(self.total_time, 5)}s."
        )
        # Add the context if required
        if self.context:
            log_str += f" Context: {self.context}"

        # Log at required level, only INFO and DEBUG supported for now
        if self._log_level == logging.INFO:
            _log.info(log_str)
        elif self._log_level == logging.DEBUG:
            _log.debug(log_str)
        else:
            raise ValueError(f"Unsupported log level {self._log_level}")

        metrics_logger.add_metric(
            TimeMetric(self.name, self.total_time, context=self.context)
        )

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def timed(
    name: str, context: Optional[Dict[str, str]] = None, log_level=logging.INFO
):
    """ Wrap an operation in this to automatically add metrics """
    return TimedOperation(name, context, log_level)
