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
    df_inputs = pd.read_csv(os.path.join(data_base_path, f"{data_version}_declare_only", task_name,
                                         f"{data_split}.tsv"), sep="\t")

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


########################  moral acceptability/agreement text ########################
def get_gold_single_input_task_text(data_version, task_name, data_split):
    data_base_path = os.path.join(PROJECT_ROOT, "data")
    df_inputs = pd.read_csv(os.path.join(data_base_path, f"{data_version}_declare_only", task_name,
                                         f"{data_split}.tsv"), sep="\t")
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


########################  main ########################
def main_get_accuracy(results_dir, data_version, data_split, check_points=None,
                      is_include_accept_class=True,
                      is_include_accept_text=True,
                      is_include_agree_class=True,
                      is_include_agree_text=True):
    eval_dir = os.path.join(results_dir, f"{data_split}_eval")

    print("=" * 20)

    if check_points is None:
        check_points = get_check_points(eval_dir)[1:]

    for check_point in check_points:
        print("=" * 40, check_point, "=" * 40)

        main_moral_acceptability(eval_dir, check_point,
                                 data_version, data_split,
                                 get_gold_single_input_task_class,
                                 get_pred_single_input_task_class,
                                 get_gold_single_input_task_text,
                                 get_pred_single_input_task_text,
                                 is_include_accept_class,
                                 is_include_accept_text)

        main_moral_agreement(eval_dir, check_point,
                             data_version, data_split,
                             get_gold_single_input_task_class,
                             get_pred_single_input_task_class,
                             get_gold_single_input_task_text,
                             get_pred_single_input_task_text,
                             is_include_agree_class,
                             is_include_agree_text)


if __name__ == "__main__":
    results_dir = os.path.join(PROJECT_ROOT, "results", "v11", "unicorn-pt", "declare_only", "lr-0.0001_bs-16")
    # check_points = [1266200]
    check_points = None
    main_get_accuracy(results_dir, "v11", "validation", check_points)
    # main_get_accuracy(results_dir, "v11", "test", check_points)
