#! /usr/bin/env python

"""
Evaluate the model checkpoint
"""

import t5
import os
import sys
import util
import seqio
import click
import logging

import tasks, mixtures
# N.B. We must import tasks and mixtures here so that they are registered and available for evaluation.

logger = logging.getLogger(__name__)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../.."))


@click.command()
@click.argument("mixture", type=str)
@click.argument("results_dir", type=str)
@click.argument("split", type=str)
@click.argument("checkpoint", type=int)
@click.option(
    "--batch-size",
    type=int,
    default=16,
    help="The batch size for evaluation. Defaults to 16.",
)

def evaluate(
    mixture: str,
    results_dir: str,
    split: str,
    checkpoint: int,
    batch_size: int,
) -> None:
    """
    Evaluate the model located at RESULTS_DIR on MIXTURE.
    """

    # Validate arguments
    util.validate_path(results_dir)

    checkpoints = util.get_result_check_points(results_dir, split, "ethics_cm_converted_class_only")

    print("-" * 10, "checkpoints todo", "-" * 10)

    if checkpoint == 100:
        checkpoints_to_eval = None
    elif checkpoint == 0:
        checkpoints_to_eval = checkpoints
    else:
        checkpoints_to_eval = [checkpoint]
    print(checkpoints_to_eval)

    # Run evaluation
    model = t5.models.HfPyTorchModel(
        model_spec=results_dir,
        model_dir=results_dir,
        batch_size=batch_size,
    )

    model.eval(
        mixture_or_task_name=mixture,
        checkpoint_steps=checkpoints_to_eval,
        split=split,
        sequence_length={"inputs": 512, "targets": 128},
    )


if __name__ == "__main__":
    evaluate()
