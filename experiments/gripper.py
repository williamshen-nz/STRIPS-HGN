from default_args import get_training_args, DomainAndProblemConfiguration
from train import train_wrapper

_CONFIGURATION = DomainAndProblemConfiguration(
    base_directory="../benchmarks/gripper",
    domain_pddl="domain.pddl",
    # {1, 2, 3 balls} = 3 problems
    problem_pddls=[
        "problems/gripper-n1.pddl",
        "problems/gripper-n2.pddl",
        "problems/gripper-n3.pddl",
    ],
)
assert len(_CONFIGURATION.problems) == 3


if __name__ == "__main__":
    train_wrapper(
        args=get_training_args(
            configurations=[_CONFIGURATION],
            # 90 seconds
            max_training_time=90,
            num_bins=3,
        )
    )
