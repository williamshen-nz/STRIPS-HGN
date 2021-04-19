import argparse
from dataclasses import dataclass
from typing import List, Optional

from strips_hgn.planning import STRIPSProblem
from strips_hgn.planning.utils import generate_strips_problems


@dataclass(frozen=True)
class BaseArgs(object):

    # Domain and problem PDDLs
    domain: Optional[str]
    domains: Optional[List[str]]
    problems: List[str]

    # Debug mode
    debug: bool

    def validate(self):
        """ Validate args for base parser """
        assert len(self.problems) == len(set(self.problems)), (
            "Please ensure a problem is only specified at most once using "
            "the '-p' or '--problems' flag."
        )
        if self.domains:
            # Check the number of domains matches the number of problems
            assert len(self.domains) == len(self.problems), (
                "Please ensure you have the same number of domains and "
                "problems when using the '-D' or '--domains' flag."
            )

    def get_strips_problems(self) -> List[STRIPSProblem]:
        return generate_strips_problems(
            domain_pddl=self.domain,
            domain_pddls=self.domains,
            problem_pddls=self.problems,
        )


def str2bool(v):
    """ https://stackoverflow.com/a/43357954 """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_base_parser(parser_description, show_defaults=True):
    """ Base parser that takes PDDL domain and problems """
    parser = argparse.ArgumentParser(description=parser_description)
    if show_defaults:
        parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter

    # PDDL domain(s)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-d", "--domain", help="Domain PDDL")
    group.add_argument(
        "-D",
        "--domains",
        nargs="+",
        help="Domain PDDLs. Will be matched against each element in Problem "
        "PDDLs (-p, --problems) flag. Use this instead of the (-d, --domain) "
        "flag where multiple domains are being specified.",
    )

    # PDDL problems
    parser.add_argument(
        "-p", "--problems", nargs="+", help="Problem PDDLs", required=True
    )

    # Debug logger
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug level logging"
    )

    return parser
