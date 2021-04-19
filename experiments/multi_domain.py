from default_args import get_training_args, DomainAndProblemConfiguration
from train import train_wrapper

_BLOCKSWORLD_CONFIGURATION = DomainAndProblemConfiguration(
    base_directory="../benchmarks/blocks-slaney",
    domain_pddl="domain.pddl",
    # 5 x {4, 5 blocks} = 10 BW problems
    problem_pddls=[
        "blocks4/task01.pddl",
        "blocks4/task02.pddl",
        "blocks4/task03.pddl",
        "blocks4/task04.pddl",
        "blocks4/task05.pddl",
        "blocks5/task01.pddl",
        "blocks5/task02.pddl",
        "blocks5/task03.pddl",
        "blocks5/task04.pddl",
        "blocks5/task05.pddl",
    ],
)
assert len(_BLOCKSWORLD_CONFIGURATION.problems) == 10

_ZENOTRAVEL_CONFIGURATION = DomainAndProblemConfiguration(
    base_directory="../benchmarks/zenotravel",
    domain_pddl="domain.pddl",
    # 5 x {2, 3 cities} = 10 Zenotravel problems
    problem_pddls=[
        "train/zenotravel-cities2-planes1-people3-8798.pddl",
        "train/zenotravel-cities2-planes2-people3-9145.pddl",
        "train/zenotravel-cities2-planes3-people3-3417.pddl",
        "train/zenotravel-cities2-planes4-people2-4892.pddl",
        "train/zenotravel-cities2-planes4-people4-6874.pddl",
        "train/zenotravel-cities3-planes1-people3-4791.pddl",
        "train/zenotravel-cities3-planes2-people3-8752.pddl",
        "train/zenotravel-cities3-planes2-people5-7306.pddl",
        "train/zenotravel-cities3-planes3-people3-1826.pddl",
        "train/zenotravel-cities3-planes3-people5-4582.pddl",
    ],
)
assert len(_ZENOTRAVEL_CONFIGURATION.problems) == 10

_GRIPPER_CONFIGURATION = DomainAndProblemConfiguration(
    base_directory="../benchmarks/gripper",
    domain_pddl="domain.pddl",
    # First 3 gripper probs
    problem_pddls=[
        "problems/gripper-n1.pddl",
        "problems/gripper-n2.pddl",
        "problems/gripper-n3.pddl",
    ],
)
assert len(_GRIPPER_CONFIGURATION.problems) == 3


if __name__ == "__main__":
    train_wrapper(
        args=get_training_args(
            configurations=[
                _BLOCKSWORLD_CONFIGURATION,
                _ZENOTRAVEL_CONFIGURATION,
                _GRIPPER_CONFIGURATION,
            ],
            # 15 minutes
            max_training_time=15 * 60,
        )
    )
