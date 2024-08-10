import argparse

def parse_args():
    """
    Arguments parser.
    """
    command_parser = argparse.ArgumentParser(
        description="Description",
        epilog="Example: "
    )

    command_parser.add_argument(
        "dataset",
        help="Dataset",
        type=str,
        choices=["muta", "pyrimidine", "triazine", "yeast", "bupa", "dummy"]
    )
    command_parser.add_argument(
        "-filename",
        help="Program to analyse",
        type=str
    )
    command_parser.add_argument(
        "-pr",
        "--prob-rules",
        help="Add probabilities to rules",
        action="store_true"
    )
    command_parser.add_argument(
        "-i",
        "--with-id",
        help="Use arity +1 where first argument is the ID of the example",
        action="store_true"
    )
    command_parser.add_argument(
        "--scores",
        help="Computes AUC and ROC scores",
        action="store_true"
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
        "-e",
        "--epochs",
        help="Number of epochs.",
        type=int,
        default=1_000
    )
    command_parser.add_argument(
        "-lr",
        help="Learning rate.",
        type=float,
        default=0.1
    )
    command_parser.add_argument(
        "-l2",
        help="Weight decay (L2 penalty).",
        type=float,
        default=0
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

