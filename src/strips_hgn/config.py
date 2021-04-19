import logging

"""
This module specifies some default configuration settings for the whole
training workflow.
"""

# Log level for the logger in the strips_hgn/training_data module

TRAINING_DATA_TIMER_LOG_LEVEL = logging.DEBUG

# Maximum time to run Fast Downward when solving training problems optimally
MAX_FD_SEARCH_TIME = 120


# For 'get_kfold_training_data'
DEFAULT_NUM_FOLDS = 10
DEFAULT_NUM_BINS = 4

DOMAIN_TO_NUM_BINS_OVERRIDE = {"gripper-strips": 3, "hanoi": 5, "sokoban": 5}
DEFAULT_DOMAIN_TO_MIN_SAMPLES = {"gripper-strips": 60, "hanoi": 50}

# Training defaults
DEFAULT_HIDDEN_SIZE = 32
