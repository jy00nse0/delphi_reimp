import os
import sys
sys.path.append("script/evaluate")
from evaluate_utils import *

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../.."))


# ######################## moral acceptability/agreement class ########################
def get_gold_single_input_task_class(data_version, task_name, data_split):
    """
    Get gold inputs and targets class labels
    """
    data_base_path = os.path.join(PROJECT_ROOT, "data")
    df_inputs = pd.read_csv(os.path.join(data_base_path, f"{data_version}_maj_vote", task_name,
                                         f"{data_split}.{task_name}.tsv"), sep="\t")

    inputs_all = list(df_inputs["inputs"])
    inputs = [i.split("[moral_single]: ")[-1] for i in inputs_all]

    targets_all = list(df_inputs["targets"])
    targets = [int(i.split("[/class] [text]")[0].split("[class]")[-1]) for i in targets_all]
    return inputs, targets


def get_pred_single_input_task_class(base_path, task_name, check_point):
    """
    Get preds class labels from local prediction file.
    """
    pred_path = os.path.join(base_path, f"{task_name}_{check_point}_predictions")
    with open(pred_path, "r") as f:
        preds_blob_list = f.read().split("\n")[1:]

    preds_class = []
    for i in preds_blob_list:
        try:
            preds_class.append(int(i.split("[/class] [text]")[0].split("[class]")[-1]))
        except:
            print("output form not identifiable:", i)
            preds_class.append(1)
    return preds_class


def get_gold_single_input_task_class_wild_v11(data_version, task_name, data_split):
    data_base_path = os.path.join(PROJECT_ROOT, "data")

    if data_split == "validation":
        df_inputs = pd.read_csv(os.path.join(data_base_path, f"{data_version}_maj_vote", "wild",
                                             "dev.tsv"), sep="\t")
    elif data_split == "test":
        df_inputs = pd.read_csv(os.path.join(data_base_path, f"{data_version}_maj_vote", "wild",
                                             f"{task_name}.tsv"), sep="\t")
    else:
        print("ERROR: not validation split")

    inputs_all = list(df_inputs["inputs"])
    targets_all = list(df_inputs["targets"])

    inputs = []
    for _, i in enumerate(inputs_all):
        if type(i) != type(""):
            print("gold class input error:", _, i, targets_all[_])
            inputs.append("")
        else:
            inputs.append(i.split("[moral_single]: ")[-1])

    targets = []
    for _, i in enumerate(targets_all):
        if type(i) != type(""):
            print("gold class output error:", _, i, inputs_all[_])
            targets.append(0)
        else:
            targets.append(int(i.split("[/class] [text]")[0].split("[class]")[-1]))

    return inputs, targets


def get_pred_single_input_task_class_wild_v11(base_path, task_name, check_point):
    if task_name == "general_test":
        pred_path = os.path.join(base_path, f"wild_{check_point}_predictions")
    else:
        pred_path = os.path.join(base_path, f"{task_name}_{check_point}_predictions")
    with open(pred_path, "r") as f:
        preds_blob_list = f.read().split("\n")[1:]

    preds_class = []
    for i in preds_blob_list:
        try:
            preds_class.append(int(i.split("[/class] [text]")[0].split("[class]")[-1]))
        except:
            print("output form not identifiable:", i)
            preds_class.append(1)
    return preds_class


########################  moral acceptability/agreement text ########################
def get_gold_single_input_task_text(data_version, task_name, data_split):
    data_base_path = os.path.join(PROJECT_ROOT, "data")
    df_inputs = pd.read_csv(os.path.join(data_base_path, f"{data_version}_maj_vote", task_name,
                                         f"{data_split}.{task_name}.tsv"), sep="\t")
    inputs_all = list(df_inputs["inputs"])
    inputs = [s.split("[moral_single]: ")[-1] for s in inputs_all]

    targets_all = list(df_inputs["targets"])
    targets = [i.split("[/class] [text]")[1].split("[/text]")[0] for i in targets_all]
    return inputs, targets


def get_pred_single_input_task_text(base_path, task_name, check_point):
    pred_path = os.path.join(base_path, f"{task_name}_{check_point}_predictions")
    with open(pred_path, "r") as f:
        preds_blob_list = f.read().split("\n")[1:]

    preds_text = []
    for i in preds_blob_list:
        try:
            preds_text.append(i.split("[/class] [text]")[1].split("[/text")[0])
        except:
            print("output form not identifiable:", i)
            preds_text.append("")
    return preds_text


def get_gold_single_input_task_text_wild_v11(data_version, task_name, data_split):
    data_base_path = os.path.join(PROJECT_ROOT, "data")
    if data_split == "validation":
        df_inputs = pd.read_csv(os.path.join(data_base_path, f"{data_version}_maj_vote", "wild", "dev.tsv"), sep="\t")
    elif data_split == "test":
        df_inputs = pd.read_csv(os.path.join(data_base_path, f"{data_version}_maj_vote", "wild",
                                             f"{task_name}.tsv"), sep="\t")
    else:
        print("ERROR: not validation split")

    inputs_all = list(df_inputs["inputs"])
    targets_all = list(df_inputs["targets"])

    inputs = []
    for _, i in enumerate(inputs_all):
        if type(i) != type(""):
            print("gold text input error:", _, i, targets_all[_])
            inputs.append("")
        else:
            inputs.append(i.split("[moral_single]: ")[-1])

    targets = []
    for _, i in enumerate(targets_all):
        if type(i) != type(""):
            print("gold text output error:", _, i, inputs_all[_])
            targets.append(0)
        else:
            targets.append(i.split("[/class] [text]")[1].split("[/text]")[0])

    return inputs, targets


def get_pred_single_input_task_text_wild_v11(base_path, task_name, check_point):
    if task_name == "general_test":
        pred_path = os.path.join(base_path, f"wild_{check_point}_predictions")
    else:
        pred_path = os.path.join(base_path, f"{task_name}_{check_point}_predictions")
    with open(pred_path, "r") as f:
        preds_blob_list = f.read().split("\n")[1:]

    preds_text = []
    for i in preds_blob_list:
        try:
            preds_text.append(i.split("[/class] [text]")[1].split("[/text")[0])
        except:
            print("output form not identifiable:", i)
            preds_text.append("")
    return preds_text


########################  main ########################
def main_get_accuracy(results_dir, data_version, data_split, check_points=None):
    eval_dir = os.path.join(results_dir, f"{data_split}_eval")

    if check_points is None:
        check_points = get_check_points(eval_dir)[2:]

    for check_point in check_points:
        print("=" * 40, check_point, "=" * 40)

        main_wild_v11(eval_dir, check_point,
                      data_version, data_split,
                      get_gold_single_input_task_class_wild_v11,
                      get_pred_single_input_task_class_wild_v11,
                      get_gold_single_input_task_text_wild_v11,
                      get_pred_single_input_task_text_wild_v11, "general_test")

        if data_split == "test":
            main_wild_v11(eval_dir, check_point,
                          data_version, data_split,
                          get_gold_single_input_task_class_wild_v11,
                          get_pred_single_input_task_class_wild_v11,
                          get_gold_single_input_task_text_wild_v11,
                          get_pred_single_input_task_text_wild_v11, "race_test")

            main_wild_v11(eval_dir, check_point,
                          data_version, data_split,
                          get_gold_single_input_task_class_wild_v11,
                          get_pred_single_input_task_class_wild_v11,
                          get_gold_single_input_task_text_wild_v11,
                          get_pred_single_input_task_text_wild_v11, "gender_test")


if __name__ == "__main__":
    results_dir = os.path.join(PROJECT_ROOT, "results", "v11", "unicorn-pt", "distribution", "lr-0.0001_bs-16")
    check_points = None
    # main_get_accuracy(results_dir, "v11", "validation", check_points)
    main_get_accuracy(results_dir, "v11", "test", [1249400])
