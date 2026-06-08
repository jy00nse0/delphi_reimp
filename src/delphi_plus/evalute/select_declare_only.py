import os
import sys
sys.path.append("script/evaluate")
from evaluate_utils import *

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../.."))


def eval_accept(row_accuracies, df_results):
    class_targets = df_results["freeform_class_targets"].tolist()
    class_preds = df_results["freeform_class_preds"].tolist()
    row_accuracies.append(get_accuracy(class_targets, class_preds, accuracy_type="exact"))
    row_accuracies.append(get_accuracy(class_targets, class_preds, accuracy_type="binary"))

    text_class_targets = df_results["moral_acceptability_text_2_class_targets"].tolist()
    text_class_preds = df_results["moral_acceptability_text_2_class_preds"].tolist()
    row_accuracies.append(get_accuracy(text_class_targets, text_class_preds, accuracy_type="binary"))

    text_targets = df_results["moral_acceptability_text_targets"].tolist()
    text_preds = df_results["moral_acceptability_text_preds"].tolist()
    exact_match_accuracy = get_moral_acceptability_text_exact_match_accuracy(text_targets, text_preds)
    return row_accuracies


def eval_agree(row_accuracies, df_results):
    class_targets = df_results["yesno_class_targets"].tolist()
    class_preds = df_results["yesno_class_preds"].tolist()
    row_accuracies.append(get_accuracy(class_targets, class_preds, accuracy_type="binary"))

    text_targets = df_results["moral_agreement_text_targets"].tolist()
    text_preds = df_results["moral_agreement_text_preds"].tolist()
    exact_match_accuracy, polarity_align_accuracy = get_moral_agreement_text_accuracy(text_targets, text_preds)
    row_accuracies.append(polarity_align_accuracy)
    return row_accuracies


def select_check_point(data_split, model_type, pt_model, bs, check_points):
    data_version = "v11"
    lr = 0.0001

    print("model_type:", model_type)
    print("lr:", lr)
    print("bs:", bs)

    result_dir = os.path.join(PROJECT_ROOT, "results", data_version,
                              pt_model, model_type, f"lr-{lr}_bs-{bs}",
                              "freeform", data_split)

    if check_points is None:
        check_points = get_result_check_points(result_dir)[2:]

    accuracies = []
    for check_point in check_points:
        row_accuracies = [check_point]

        ##################### accept #####################
        df_results = read_result_file(data_version, model_type,
                                      check_point, data_split, "freeform", lr, bs, pt_model)
        row_accuracies = eval_accept(row_accuracies, df_results)

        ##################### agree #####################
        df_results = read_result_file(data_version, model_type,
                                      check_point, data_split, "yesno", lr, bs, pt_model)
        row_accuracies = eval_agree(row_accuracies, df_results)

        accuracies.append(row_accuracies)
        print("-- check point:", check_point, row_accuracies)

        df_to_save = pd.DataFrame(accuracies)
        df_to_save.to_csv("temp_result_file_2.csv", index=False)


if __name__ == "__main__":
    model_type = "declare_only"
    pt_model = "unicorn-pt"
    bs = 16
    check_points = None
    select_check_point("validation", model_type, pt_model, bs, check_points)
