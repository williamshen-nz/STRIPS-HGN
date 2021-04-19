import json
import logging
import os
from abc import ABC
from datetime import datetime
from enum import Enum
from json import JSONEncoder
from typing import Dict, List, Optional, Union

from strips_hgn.utils import Number

_log = logging.getLogger(__name__)


class MetricUnit(Enum):
    Seconds = "seconds"
    Count = "count"


class Metric(ABC):
    def __init__(
        self,
        name: str,
        value: Number,
        unit: MetricUnit,
        timestamp: datetime = None,
        context: Optional[Dict[str, Union[str, Number]]] = None,
    ):
        self.name = name
        self.value = value
        self.unit = unit
        self.timestamp = timestamp if timestamp else datetime.now()
        self.context = context if context else {}

    def get_metrics_dict(self) -> dict:
        return {
            self.name: {
                "value": self.value,
                "unit": self.unit.value,
                "context": self.context,
                "timestamp": self.timestamp,
            }
        }

    def to_json(self) -> str:
        return json.dumps(self.get_metrics_dict())


class TimeMetric(Metric):
    def __init__(
        self,
        name: str,
        value: Number,
        timestamp: datetime = None,
        context: Optional[Dict[str, Union[str, Number]]] = None,
    ):
        super(TimeMetric, self).__init__(
            name, value, MetricUnit.Seconds, timestamp, context
        )


class CountMetric(Metric):
    def __init__(
        self,
        name: str,
        value: Number,
        timestamp: datetime = None,
        context: Optional[Dict[str, Union[str, Number]]] = None,
    ):
        super(CountMetric, self).__init__(
            name, value, MetricUnit.Count, timestamp, context
        )


class _MetricsEncoder(JSONEncoder):
    def default(self, o):
        if isinstance(o, Metric):
            return o.get_metrics_dict()
        elif isinstance(o, datetime):
            return str(o)
        else:
            super(_MetricsEncoder, self).default(o)


class MetricsLogger(object):
    """
    Stores metrics
    """

    _DEFAULT_FILE_NAME = "metrics.json"

    def __init__(self):
        self.metrics: List[Metric] = []

    def clear(self):
        """ Remove all metrics from the Metrics Logger """
        self.metrics = []

    def add_metric(self, metric: Metric):
        self.metrics.append(metric)

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.metrics, cls=_MetricsEncoder, **kwargs)

    def save_json(
        self, file_dir: str, file_name: str = _DEFAULT_FILE_NAME, **json_kwargs
    ):
        """
        Save metrics to a file as a JSON

        Parameters
        ----------
        file_dir: str, directory to save the file
        file_name: str, name of the file
        json_kwargs: any parameters to pass to json.dump

        Returns
        -------
        str, path to the JSON metrics file
        """
        file_path = os.path.join(file_dir, file_name)
        json.dump(
            self.metrics,
            open(file_path, "w"),
            cls=_MetricsEncoder,
            **json_kwargs,
        )
        _log.info(f"Saved MetricsLogger JSON to {file_path}")
        return file_path


# Global metrics logger
metrics_logger = MetricsLogger()
