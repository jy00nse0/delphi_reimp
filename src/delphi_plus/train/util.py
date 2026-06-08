"""
Util functions for fine-tuning and evaluating models
"""
import os
import seqio
import pandas as pd
import tensorflow_datasets as tfds
import tensorflow as tf


def print_task_examples(task_name, split="validation", num_ex=1):
    """
    Print examples from tasks
    """
    print("#" * 20, task_name, "#" * 20)
    task = seqio.TaskRegistry.get(task_name)
    ds = task.get_dataset(split=split, sequence_length={"inputs": 512, "targets": 128})
    for i, ex in enumerate(tfds.as_numpy(ds.take(num_ex))):
        print(i, ex)
    print("test", task.num_input_examples("test"))
    print("train", task.num_input_examples("train"))
    print("validation", task.num_input_examples("validation"))


def print_mixture_examples(mixture_name, split="validation", num_ex=1):
    """
    Print examples from mixtures
    """
    print("#" * 20, mixture_name, "#" * 20)
    mixture = seqio.MixtureRegistry.get(mixture_name)
    ds = mixture.get_dataset(split=split,
                             sequence_length={"inputs": 512, "targets": 128})

    for i, ex in enumerate(tfds.as_numpy(ds.take(num_ex))):
        print(i, ex)
    print("test", mixture.num_input_examples("test"))
    print("train", mixture.num_input_examples("train"))
    print("validation", mixture.num_input_examples("validation"))


def get_num_elements_csv(file_name):
    """
    Get the total number of elements in a given csv/tsv file
    """
    df = pd.read_csv(file_name, delimiter="\t", header=None)
    return df.shape[0]


def get_num_elements_split(split_paths):
    """
    Get the number of elements in each split of a dataset
    """
    num_elements_split = {}
    for split, path in split_paths.items():
        num_elements_split[split] = get_num_elements_csv(path)
    return num_elements_split


def get_result_check_points(result_prefix, split, eval_data_type, after_check_point=-1):
    """
    Get a list of model checkpoints that haven't generated predictions on the
    designated data split yet. Supports HuggingFace-style (checkpoint-{N})
    and TensorFlow-style ({N}.meta) checkpoint formats.
    """
    check_points = []
    done_check_points = []

    if os.path.exists(result_prefix):
        for fname in os.listdir(result_prefix):
            full_path = os.path.join(result_prefix, fname)
            # HuggingFace checkpoint directories: checkpoint-{step}
            if os.path.isdir(full_path) and fname.startswith("checkpoint-"):
                try:
                    check_point = int(fname.split("checkpoint-")[-1])
                    if check_point > after_check_point:
                        check_points.append(check_point)
                except ValueError:
                    pass
            # TensorFlow checkpoint files: model.ckpt-{step}.meta
            elif fname.endswith(".meta"):
                try:
                    check_point = int(fname.split(".meta")[0].split("-")[-1])
                    if check_point > after_check_point:
                        check_points.append(check_point)
                except ValueError:
                    pass

    print("-" * 10, "checkpoints all", "-" * 10)
    print(check_points)

    eval_dir = os.path.join(result_prefix, f"{split}_eval")
    if os.path.exists(eval_dir):
        for fname in os.listdir(eval_dir):
            if "_predictions" in fname and eval_data_type in fname and "_predictions_clean" not in fname:
                try:
                    check_point_done = int(fname.split("_predictions")[0].split("_")[-1])
                    if check_point_done in check_points:
                        done_check_points.append(check_point_done)
                        check_points.remove(check_point_done)
                except ValueError:
                    pass

    print("-" * 10, "checkpoints done", "-" * 10)
    print(done_check_points)
    return check_points


def validate_path(results_dir, pretrained_model=None, PRETRAINED_MODELS=None):
    """
    Validate local result path
    """
    if PRETRAINED_MODELS is not None:
        parent_dir = os.path.dirname(os.path.abspath(results_dir))
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)

        if pretrained_model not in PRETRAINED_MODELS:
            if not os.path.exists(str(pretrained_model)):
                raise IOError(
                    f"--pretrained-model ({pretrained_model}) does not exist."
                    f" It must be a valid local path or one of"
                    f' {", ".join(PRETRAINED_MODELS.keys())}.')
    else:
        if not os.path.exists(results_dir):
            raise IOError(f"RESULTS_DIR ({results_dir}) doesn't exist.")


def print_arguments(result_path, results_dir, mixture, split, pretrained_model,
                    n_steps, batch_size, save_checkpoints_steps, n_checkpoints_to_keep,
                    learning_rate, tasks, continue_finetune):
    print("=" * 10, "results_dir")
    print(results_dir)

    print("=" * 10, "mixture")
    print(mixture)

    print("=" * 10, "split")
    print(split)

    print("=" * 10, "pretrained_model")
    print(pretrained_model)

    print("=" * 10, "n_steps")
    print(n_steps)

    print("=" * 10, "batch_size")
    print(batch_size)

    print("=" * 10, "save_checkpoints_steps")
    print(save_checkpoints_steps)

    print("=" * 10, "n_checkpoints_to_keep")
    print(n_checkpoints_to_keep)

    print("=" * 10, "learning_rate")
    print(learning_rate)

    print("=" * 10, "result_path")
    print(result_path)

    print("=" * 10, "data_version")
    print(tasks.data_version)

    print("=" * 10, "continue_finetune")
    print(continue_finetune)


def get_result_path(
        results_dir: str,
        pretrained_model: str,
        mixture: str,
        learning_rate: float,
        batch_size: int
) -> str:
    """
    Get a result path given arguments
    """
    result_path = os.path.join(
        results_dir,
        pretrained_model,
        mixture,
        f"lr-{learning_rate}_bs-{batch_size}"
    )
    return result_path
