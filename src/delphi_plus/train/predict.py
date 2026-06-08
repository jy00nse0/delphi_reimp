#! /usr/bin/env python

"""Predict using the fine-tuned model."""

import t5
import os
import sys
import seqio
import logging
import click
import util
import pandas as pd

logger = logging.getLogger(__name__)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../.."))


@click.command()
@click.option(
    "--batch-size",
    type=int,
    default=64,
    help="The batch size to use for prediction. Defaults to 64.",
)

def predict(
    batch_size: int,
) -> None:
    """Run prediction using the local fine-tuned model."""

    eval_data = "race_topk_batch6to10"

    data_version = "v9"
    model_type = "sbic_commonsense_morality_joint_all_proportional"
    check_point = 1264700
    lr = 0.0001
    bs = 16
    training_type = model_type.split("_")[-3]

    models_dir = os.path.join(PROJECT_ROOT, "model", data_version, "unicorn-pt",
                              model_type, f"lr-{lr}_bs-{bs}")

    # Run prediction.
    model = t5.models.HfPyTorchModel(
        model_spec=models_dir,
        model_dir=models_dir,
        batch_size=batch_size,
    )

    predict_joint_inputs_paths = [os.path.join(
        PROJECT_ROOT, "data", "qualitative_eval", training_type,
        eval_data + "_qualitative_eval.tsv"
    )]
    predict_joint_outputs_paths = [os.path.join(
        PROJECT_ROOT, "preds", data_version, "unicorn-pt",
        model_type, f"lr-{lr}_bs-{bs}", "raw",
        eval_data + "_qualitative_eval.tsv"
    )]

    for i in range(len(predict_joint_inputs_paths)):
        predict_joint_inputs_path = predict_joint_inputs_paths[i]
        predict_joint_outputs_path = predict_joint_outputs_paths[i]
        os.makedirs(os.path.dirname(predict_joint_outputs_path), exist_ok=True)

        model.predict(
            input_file=predict_joint_inputs_path,
            output_file=predict_joint_outputs_path,
            temperature=0,
            checkpoint_steps=check_point,
        )


if __name__ == "__main__":
    predict()
