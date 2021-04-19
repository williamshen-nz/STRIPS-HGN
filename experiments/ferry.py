from default_args import get_training_args, DomainAndProblemConfiguration
from train import train_wrapper

_CONFIGURATION = DomainAndProblemConfiguration(
    base_directory="../benchmarks/ferry",
    domain_pddl="ferry.pddl",
    # {2, 3, 4 locations} x {1, 2, 3} cars = 9 problems
    problem_pddls=[
        "train/ferry-l2-c1.pddl",
        "train/ferry-l2-c2.pddl",
        "train/ferry-l2-c3.pddl",
        "train/ferry-l3-c1.pddl",
        "train/ferry-l3-c2.pddl",
        "train/ferry-l3-c3.pddl",
        "train/ferry-l4-c1.pddl",
        "train/ferry-l4-c2.pddl",
        "train/ferry-l4-c3.pddl",
    ],
)
assert len(_CONFIGURATION.problems) == 9


if __name__ == "__main__":
    train_wrapper(
        args=get_training_args(
            configurations=[_CONFIGURATION],
            # 3 minutes
            max_training_time=3 * 60,
            num_folds=5,
        )
    )
