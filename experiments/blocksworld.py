from default_args import get_training_args, DomainAndProblemConfiguration
from train import train_wrapper

_CONFIGURATION = DomainAndProblemConfiguration(
    base_directory="../benchmarks/blocks-slaney",
    domain_pddl="domain.pddl",
    # {3, 4, 5 blocks} x 10 problems = 30 problems
    problem_pddls=[
        "blocks3/task01.pddl",
        "blocks3/task02.pddl",
        "blocks3/task03.pddl",
        "blocks3/task04.pddl",
        "blocks3/task05.pddl",
        "blocks3/task06.pddl",
        "blocks3/task07.pddl",
        "blocks3/task08.pddl",
        "blocks3/task09.pddl",
        "blocks3/task10.pddl",
        "blocks4/task01.pddl",
        "blocks4/task02.pddl",
        "blocks4/task03.pddl",
        "blocks4/task04.pddl",
        "blocks4/task05.pddl",
        "blocks4/task06.pddl",
        "blocks4/task07.pddl",
        "blocks4/task08.pddl",
        "blocks4/task09.pddl",
        "blocks4/task10.pddl",
        "blocks5/task01.pddl",
        "blocks5/task02.pddl",
        "blocks5/task03.pddl",
        "blocks5/task04.pddl",
        "blocks5/task05.pddl",
        "blocks5/task06.pddl",
        "blocks5/task07.pddl",
        "blocks5/task08.pddl",
        "blocks5/task09.pddl",
        "blocks5/task10.pddl",
    ],
)
assert len(_CONFIGURATION.problems) == 30


if __name__ == "__main__":
    train_wrapper(
        args=get_training_args(
            configurations=[_CONFIGURATION],
            # 10 minutes
            max_training_time=10 * 60,
        )
    )
