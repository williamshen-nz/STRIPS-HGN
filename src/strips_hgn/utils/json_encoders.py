from enum import Enum
from json import JSONEncoder
from typing import Any

from strips_hgn.features import AbstractFeatureMapper


class MetricsEncoder(JSONEncoder):
    """ For encoding evaluation metrics """

    def default(self, o: Any) -> Any:
        if isinstance(o, Enum):
            return o.value
        else:
            return super().default(o)


class ArgsEncoder(JSONEncoder):
    """ For encoding command line arguments """

    def default(self, o: Any) -> Any:
        if issubclass(o, AbstractFeatureMapper):
            return o.name()
        elif isinstance(o, Enum):
            return o.value
        else:
            return super().default(o)
