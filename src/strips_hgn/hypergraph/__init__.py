from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Optional, TypeVar

from strips_hgn.utils import Number

# Node in the Hypergraph
Node = TypeVar("Node", bound=str)


@dataclass(frozen=True)
class Hyperedge(object):
    """
    Simple class representing a directed hyperedge.
    - `senders` represent the nodes in the tail of the hyperedge,
    - `receivers` represent the nodes in the head of the hyperedge
    """

    name: str
    weight: Number
    senders: FrozenSet[Node]
    receivers: FrozenSet[Node]

    # Additional context which may be required for feature mappers, etc.
    context: Optional[Dict[str, Any]] = field(default=None, compare=False)
