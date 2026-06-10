#! /usr/bin/env python

"""Fine-tune T5 based models."""

import os
import click
import logging
import warnings
import pandas as pd
import torch
from torch.utils.data import Dataset, ConcatDataset
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from transformers.trainer_utils import get_last_checkpoint
import util

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=DeprecationWarning)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../.."))

PRETRAINED_MODELS = {
    "small": "t5-small",
    "base": "t5-base",
    "large": "t5-large",
    "3B": "t5-3b",
    "11B": "t5-11b",
    "unicorn-pt": os.path.join(PROJECT_ROOT, "models", "unicorn-pt"),
    "v11-delphi-declare": os.path.join(PROJECT_ROOT, "models", "v11-delphi-declare"),
}

# Mixture → list of data directories (each directory has {split}.tsv or {split}.{task}.tsv)
MIXTURE_DATA = {
    "declare_only": [
        os.path.join(PROJECT_ROOT, "data", "v11_declare_only", "freeform"),
        os.path.join(PROJECT_ROOT, "data", "v11_declare_only", "yesno"),
    ],
    "norm_bank": [
        os.path.join(PROJECT_ROOT, "data", "v11_maj_vote", "moral_acceptability"),
        os.path.join(PROJECT_ROOT, "data", "v11_maj_vote", "moral_agreement"),
        os.path.join(PROJECT_ROOT, "data", "v11_maj_vote", "moral_comparison"),
    ],
}


class T5TSVDataset(Dataset):
    """Loads a NORM BANK TSV file and builds Delphi's text-to-text I/O format.

    The input is given a task specifier prefix ('[moral_single]:' for free-form /
    yes/no), and the target jointly encodes the classification label and the
    open-text judgment wrapped in bracket tags (Delphi+ style, consistent with the
    prefix bracket form and free of T5's '<' -> <unk> tokenization issue):
        [class] {class_label} [/class] [text] {text_label} [/text]
    """

    def __init__(self, file_path, tokenizer, prefix="[moral_single]:",
                 max_input_length=512, max_target_length=128):
        df_raw = pd.read_csv(file_path, sep="\t", index_col=0)
        if "inputs" in df_raw.columns and "targets" in df_raw.columns:
            # Pre-formatted file: inputs/targets already contain prefix and tags
            self.df = df_raw[["inputs", "class", "targets"]] if "class" in df_raw.columns \
                else df_raw[["inputs", "targets"]].assign(**{"class": ""})
        else:
            # Raw NORM BANK columns: input_sequence, class_label, text_label
            self.df = df_raw[["input_sequence", "class_label", "text_label"]].rename(
                columns={"input_sequence": "inputs", "class_label": "class", "text_label": "targets"}
            )
        self.tokenizer = tokenizer
        self.prefix = prefix
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Prepend the task specifier to the input
        input_text = f"{self.prefix} {row['inputs']}"
        # Jointly encode classification label and open-text judgment
        target_text = (
            f"[class] {row['class']} [/class] [text] {row['targets']} [/text]"
        )
        input_enc = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target_enc = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        labels = target_enc["input_ids"].squeeze().clone()
        # Mask padding tokens so they are ignored in loss computation
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids": input_enc["input_ids"].squeeze(),
            "attention_mask": input_enc["attention_mask"].squeeze(),
            "labels": labels,
        }


def load_mixture_dataset(mixture, split, tokenizer):
    """Concatenate all task datasets belonging to the given mixture."""
    if mixture not in MIXTURE_DATA:
        raise ValueError(
            f"Unknown mixture '{mixture}'. Available: {list(MIXTURE_DATA.keys())}"
        )

    datasets = []
    for data_dir in MIXTURE_DATA[mixture]:
        task_name = os.path.basename(data_dir)
        # Paired (relative) tasks use '[moral_pair]:'; single tasks use '[moral_single]:'
        prefix = "[moral_pair]:" if "comparison" in task_name else "[moral_single]:"
        # Support both 'train.tsv' and 'train.{task_name}.tsv' naming conventions
        candidates = [
            os.path.join(data_dir, f"{split}.tsv"),
            os.path.join(data_dir, f"{split}.{task_name}.tsv"),
        ]
        for path in candidates:
            if os.path.exists(path):
                ds = T5TSVDataset(path, tokenizer, prefix=prefix)
                datasets.append(ds)
                print(f"Loaded: {path}  ({len(ds)} examples)")
                break
        else:
            raise FileNotFoundError(
                f"No '{split}' TSV found in {data_dir}.\nTried: {candidates}"
            )

    return ConcatDataset(datasets)


@click.command()
@click.argument("mixture", type=str)
@click.argument("results_dir", type=str)
@click.argument("pretrained-model", type=str)
@click.option(
    "--split",
    type=str,
    default="train",
    help="The split on which to train. Defaults to 'train'.",
)
@click.option(
    "--n-steps",
    type=int,
    default=600000,
    help="The number of gradient updates. Defaults to 600,000.",
)
@click.option(
    "--batch-size",
    type=int,
    default=16,
    help="The batch size for training. Defaults to 16.",
)
@click.option(
    "--save-checkpoints-steps",
    type=int,
    default=5000,
    help="Steps between checkpoint saves. Defaults to 5000.",
)
@click.option(
    "--n-checkpoints-to-keep",
    type=int,
    default=300,
    help="Maximum number of checkpoints to keep. Defaults to 300.",
)
@click.option(
    "--learning-rate",
    type=float,
    default=2e-4,
    help="The learning rate. Defaults to 2e-4.",
)
@click.option(
    "--continue_finetune",
    type=bool,
    default=True,
    help="Whether to resume from an existing checkpoint.",
)
def fine_tune(
    mixture: str,
    results_dir: str,
    pretrained_model: str,
    split: str,
    n_steps: int,
    batch_size: int,
    learning_rate: float,
    save_checkpoints_steps: int,
    n_checkpoints_to_keep: int,
    continue_finetune: bool,
) -> None:
    """Fine-tune T5 on MIXTURE, writing checkpoints to RESULTS_DIR."""

    result_path = util.get_result_path(
        results_dir, pretrained_model, mixture, learning_rate, batch_size
    )
    # Store checkpoints on the large network volume (/workspace) by default so the
    # small overlay root ('/') doesn't fill up. Absolute paths are respected as-is.
    if not os.path.isabs(result_path):
        result_path = os.path.join("/workspace", result_path)
    util.validate_path(results_dir, pretrained_model, PRETRAINED_MODELS)

    # Resolve pretrained model to its full path / HF identifier
    model_spec = PRETRAINED_MODELS.get(pretrained_model, pretrained_model)

    # Determine whether to resume from an existing checkpoint.
    # get_last_checkpoint() finds the latest valid 'checkpoint-N' subdir inside
    # result_path. The base model is still loaded from its pretrained spec below;
    # the Trainer then restores weights/optimizer/scheduler/RNG/step from this dir.
    resume_from_checkpoint = None
    if continue_finetune and os.path.isdir(result_path):
        resume_from_checkpoint = get_last_checkpoint(result_path)
        if resume_from_checkpoint is None:
            print(f"continue_finetune=True but no checkpoint in {result_path}; starting fresh.")
        else:
            print(f"Resuming from checkpoint: {resume_from_checkpoint}")

    print("=" * 10, "result_path");     print(result_path)
    print("=" * 10, "model_spec");      print(model_spec)
    print("=" * 10, "mixture");         print(mixture)
    print("=" * 10, "split");           print(split)
    print("=" * 10, "n_steps");         print(n_steps)
    print("=" * 10, "batch_size");      print(batch_size)
    print("=" * 10, "learning_rate");   print(learning_rate)

    # Load tokenizer — use original pretrained name so tokenizer files are always found
    tokenizer_spec = PRETRAINED_MODELS.get(pretrained_model, pretrained_model)
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_spec)

    # Load model
    model = T5ForConditionalGeneration.from_pretrained(model_spec)

    # Load training and validation data
    train_dataset = load_mixture_dataset(mixture, split, tokenizer)
    eval_dataset = load_mixture_dataset(mixture, "validation", tokenizer)

    # Configure training with evaluation for early stopping
    training_args = TrainingArguments(
        output_dir=result_path,
        max_steps=n_steps,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        save_steps=save_checkpoints_steps,
        save_total_limit=n_checkpoints_to_keep,
        logging_steps=100,
        report_to="none",
        # bf16 preferred for large models (avoids fp16 overflow/NaN gradients)
        # Falls back to fp16 if bf16 not supported (older GPUs like V100/T4)
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        max_grad_norm=1.0,
        # Early stopping requires periodic evaluation
        eval_strategy="steps",
        eval_steps=save_checkpoints_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)


if __name__ == "__main__":
    fine_tune()
