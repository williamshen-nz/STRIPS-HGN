from dataclasses import dataclass
from typing import Type

from strips_hgn.config import (
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_NUM_BINS,
    DEFAULT_NUM_FOLDS,
)
from strips_hgn.features import (
    GlobalFeatureMapper,
    HyperedgeFeatureMapper,
    NodeFeatureMapper,
)
from strips_hgn.features.global_features import (
    DEFAULT_GLOBAL_FEATURE_MAPPER,
    GLOBAL_FEATURE_MAPPERS,
    get_global_feature_mapper,
)
from strips_hgn.features.hyperedge_features import (
    DEFAULT_EDGE_FEATURE_MAPPER,
    EDGE_FEATURE_MAPPERS,
    get_hyperedge_feature_mapper,
)
from strips_hgn.features.node_features import (
    DEFAULT_NODE_FEATURE_MAPPER,
    NODE_FEATURE_MAPPERS,
    get_node_feature_mapper,
)
from strips_hgn.utils.args.base_args import BaseArgs, get_base_parser, str2bool


@dataclass(frozen=True)
class TrainingArgs(BaseArgs):
    """
    Fields
    ------
    num_folds: number of folds to use for k-fold
    num_bins: number of bins to split training data into before stratified
        K-Fold split
    remove_duplicates: whether to remove duplicates in training data
    shuffle: whether to shuffle training data
    global_feature_mapper_cls: feature mapper to use for mapping a hypergraph
        to its global features
    node_feature_mapper_cls: feature mapper to use for
        mapping propositions
    hyperedge_feature_mapper_cls: feature mapper to use for mapping
        actions
    batch_size: batch size to use for training
    max_training_time: maximum training time for each fold
    max_epochs: maximum number of epochs for training each fold
    patience: patience (in epochs) for early stopping
    """

    # Training data
    num_folds: int
    num_bins: int
    remove_duplicates: bool
    shuffle: bool

    # Feature mappers
    global_feature_mapper_cls: Type[GlobalFeatureMapper]
    node_feature_mapper_cls: Type[NodeFeatureMapper]
    hyperedge_feature_mapper_cls: Type[HyperedgeFeatureMapper]

    # Network Hyperparameters
    hidden_size: int

    # Training loop
    batch_size: int
    learning_rate: float
    weight_decay: float
    max_training_time: int
    max_epochs: int
    patience: int

    def validate(self):
        super().validate()
        assert len(self.problems) > 0, "Must be at least 1 problem"
        for field, threshold in (
            ("num_folds", 1),
            ("hidden_size", 0),
            ("batch_size", 0),
            ("learning_rate", 0.0),
            ("weight_decay", 0.0),
            ("max_training_time", 0),
            ("max_epochs", 0),
            ("patience", 0),
        ):
            assert (
                getattr(self, field) > threshold
            ), f"{field} must be > {threshold}, not {getattr(self, field)}"


def _get_training_parser(show_defaults=True):
    """ Generate parser for training scripts """
    parser = get_base_parser(
        "STRIPS-HGN. Training based on the perfect heuristic h*", show_defaults
    )

    parser.add_argument(
        "--global-feature",
        choices=GLOBAL_FEATURE_MAPPERS,
        default=DEFAULT_GLOBAL_FEATURE_MAPPER,
        type=lambda mapper: get_global_feature_mapper(mapper),
        help="Feature mapper to use for the global features - none = no "
        "feature mapper, num_nodes_and_edges = [|nodes|, |hyperedges|].",
        dest="global_feature_mapper_cls",
    )

    # Feature mappers
    parser.add_argument(
        "--edge-feature",
        choices=EDGE_FEATURE_MAPPERS,
        default=DEFAULT_EDGE_FEATURE_MAPPER,
        type=lambda mapper: get_hyperedge_feature_mapper(mapper),
        help="Feature mapper to use for edges (i.e. actions) - weight-only = "
        "[weight], complex = [weight, |PRE|, |EFF+|], "
        "more-complex = [weight, |PRE|, |EFF+|, |EFF-|].",
        dest="hyperedge_feature_mapper_cls",
    )

    parser.add_argument(
        "--node-feature",
        choices=NODE_FEATURE_MAPPERS,
        default=DEFAULT_NODE_FEATURE_MAPPER,
        type=lambda mapper: get_node_feature_mapper(mapper),
        help="Feature mapper to use for nodes (i.e. propositions) - simple = "
        "[x_s, x_g] where x_s = 1 (x_g = 1) iff the proposition is true in "
        "the state s (goal G), and 0 otherwise. See Section 6.1 of paper for "
        "more details.",
        dest="node_feature_mapper_cls",
    )

    parser.add_argument(
        "-f",
        "--num-folds",
        type=int,
        default=DEFAULT_NUM_FOLDS,
        help="Number of folds to split the training data into for training.",
    )

    parser.add_argument(
        "--num-bins",
        type=int,
        default=DEFAULT_NUM_BINS,
        help="Number of bins to split training data into before applying "
        "stratified K-Fold. See `config.py` for some default overrides used.",
    )

    parser.add_argument(
        "--hidden-size",
        type=int,
        default=DEFAULT_HIDDEN_SIZE,
        help="Number of hidden units in each layer for the MLPs.",
    )

    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=1,
        help="Number of samples used in each step of training",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate to use when training the STRIPS-HGN",
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=2.5e-4,
        help="Weight decay (L2 penalty) to use when training the STRIPS-HGN",
    )

    parser.add_argument(
        "--remove-duplicates",
        type=str2bool,
        default=False,
        help="Whether to remove duplicate state-value pairs for training. "
        "Specify either True (1, y, t) or False (0, n, f).",
    )

    parser.add_argument(
        "--shuffle",
        type=str2bool,
        default=True,
        help="Whether to shuffle the training data. Specify either True "
        "(1, y, t) or False (0, n, f).",
    )

    parser.add_argument(
        "-e",
        "--max-epochs",
        type=int,
        default=10000,
        help="Max number of epochs to train each fold for. If max training "
        "time is reached first, then max epochs will be ignored.",
    )

    parser.add_argument(
        "-t",
        "--max-training-time",
        type=int,
        required=True,
        help="Max training time in seconds for each fold. If max epochs is "
        "reached first, then max training time will be ignored ",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience - number of epochs for which validation "
        "loss stops decreasing in order to stop training",
    )

    return parser


def parse_and_validate_training_args() -> TrainingArgs:
    """ Parse args, validate and return TrainingArgs object """
    args = _get_training_parser().parse_args()
    args = TrainingArgs(**vars(args))
    args.validate()
    return args
