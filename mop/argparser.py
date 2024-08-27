import argparse

def parse_args():
    """
    Arguments parser.
    """
    command_parser = argparse.ArgumentParser(
        description="MOP: Mixtures Of Probabilistic logic programs",
        # epilog="Example: ",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    command_parser.add_argument(
        "-f",
        "--filename",
        help="Program to analyse",
        type=str
    )
    command_parser.add_argument(
        "-nba",
        help="Maximum number of body atoms",
        type=int,
        default=2
    )
    command_parser.add_argument(
        "-nr",
        help="maximum number of rules in each mixture.",
        type=int,
        default=2
    )
    command_parser.add_argument(
        "-nm",
        help="maximum number of mixtures to consider.",
        type=int,
        default=-1
    )
    command_parser.add_argument(
        "-maxfun",
        help="Max number of function evaluations during optimization.",
        type=int,
        default=1_000
    )
    command_parser.add_argument(
        "-gamma",
        help="Scale factor for L1 and L2.",
        type=float,
        default=0
    )
    command_parser.add_argument(
        "-l1",
        help="L1 regularization.",
        action="store_true"
    )
    command_parser.add_argument(
        "-l2",
        help="L2 regularization.",
        action="store_true"
    )
    command_parser.add_argument(
        "-cut",
        help="Cutoff value to drop mixture (i.e., drop if prob < 10e-cut).",
        type=int,
        default=10
    )
    command_parser.add_argument(
        "-v",
        "--verbosity",
        help="Verbosity level.",
        type=int,
        default=0
    )
    command_parser.add_argument(
        "--train",
        nargs="+",
        help="Folds for the training set",
        required=True
    )
    command_parser.add_argument(
        "--test",
        nargs="+",
        help="Folds for the test set"
    )

    return command_parser.parse_args()

