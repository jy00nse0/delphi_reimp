#!/usr/bin/env python
"""
통합 평가 스크립트
===================
fine-tune.py 로 저장된 HuggingFace 체크포인트에 대해
논문 Table 3 의 5 가지 지표를 계산하고 목표치와의 차이를 출력한다.

평가 지표 (모두 validation set 기준)
  Free-form C(3)  : 3-way 분류 정확도 (Positive / Discretionary / Negative)
  Free-form C(2)  : 2-way 분류 정확도 (Pos+Disc vs Neg)
  Free-form T(A)  : 개방형 텍스트 자동 평가 정확도
  Yes/no   C(2)   : 2-way 분류 정확도 (Yes vs No)
  Yes/no   T(A)   : 개방형 텍스트 자동 평가 정확도 (polarity alignment)

사용법:
  python run_eval.py RESULTS_DIR \\
      --pretrained-model large \\
      --mixture declare_only \\
      --learning-rate 0.0002 \\
      --batch-size 16 \\
      [--checkpoint 50000] \\
      [--inference-batch-size 32] \\
      [--input-prefix "[moral_single]: "]
"""

import os
import sys
import click
import pandas as pd
import torch
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer

# ── 프로젝트 루트 / util 경로 설정 ──────────────────────────────────────────
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../.."))

sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "utils"))
sys.path.insert(0, CURRENT_DIR)

from text2class import text2class, normalize_label, get_accuracy
from evaluate_utils import (
    convert_moral_acceptability_text_to_class,
    get_moral_agreement_text_accuracy,
)

# ── 논문 Table 3 목표치 ──────────────────────────────────────────────────────
PAPER_TARGETS = {
    "freeform_C(3)":   80.0,
    "freeform_C(2)":   91.5,
    "freeform_T(A)":   92.4,
    "yesno_C(2)":      97.4,
    "yesno_T(A)":      97.5,
}

METRIC_LABELS = [
    ("freeform_C(3)", "Free-form C(3)"),
    ("freeform_C(2)", "Free-form C(2)"),
    ("freeform_T(A)", "Free-form T(A)"),
    ("yesno_C(2)",    "Yes/no   C(2)"),
    ("yesno_T(A)",    "Yes/no   T(A)"),
]


# ─────────────────────────────────────────────────────────────────────────────
# 유틸리티
# ─────────────────────────────────────────────────────────────────────────────

def find_checkpoints(result_path: str):
    """result_path 내 HuggingFace checkpoint-N 디렉토리를 step 오름차순으로 반환."""
    checkpoints = []
    if not os.path.isdir(result_path):
        return checkpoints
    for name in os.listdir(result_path):
        full = os.path.join(result_path, name)
        if os.path.isdir(full) and name.startswith("checkpoint-"):
            try:
                step = int(name.split("checkpoint-")[-1])
                checkpoints.append((step, full))
            except ValueError:
                pass
    return sorted(checkpoints, key=lambda x: x[0])


def load_validation_tsv(file_path: str):
    """
    TSV 파일을 읽어 (inputs, class_labels, text_labels) 반환.
    지원 포맷 1: 컬럼 input_sequence / class_label / text_label (raw 형식)
    지원 포맷 2: 컬럼 inputs / targets  ([class]N[/class] [text]L[/text] 형식)
    """
    df = pd.read_csv(file_path, sep="\t")

    # ─ 포맷 1: raw 형식 ─
    if "input_sequence" in df.columns:
        inputs       = list(df["input_sequence"].astype(str))
        class_labels = list(df["class_label"].astype(int))
        text_labels  = list(df["text_label"].astype(str))
        return inputs, class_labels, text_labels

    # ─ 포맷 2: inputs/targets 형식 ─
    if "inputs" in df.columns and "targets" in df.columns:
        raw_inputs = list(df["inputs"].astype(str))
        raw_targets = list(df["targets"].astype(str))
        # inputs 에서 [moral_single]: 접두사 제거
        inputs = [s.split("[moral_single]: ")[-1] for s in raw_inputs]
        class_labels = [_parse_class_from_str(t) for t in raw_targets]
        text_labels  = [_parse_text_from_str(t) for t in raw_targets]
        return inputs, class_labels, text_labels

    raise ValueError(
        f"'{file_path}' 에서 지원 컬럼을 찾지 못했습니다.\n"
        f"  이용 가능한 컬럼: {list(df.columns)}"
    )


def _parse_class_from_str(s: str) -> int:
    """'[class]-1[/class] [text]...[/text]' → int"""
    try:
        return int(s.split("[/class]")[0].split("[class]")[-1])
    except Exception:
        return 99


def _parse_text_from_str(s: str) -> str:
    """'[class]...[/class] [text]label[/text]' → label"""
    try:
        return s.split("[/class] [text]")[1].split("[/text]")[0]
    except Exception:
        return s


def parse_pred_class(pred: str) -> int:
    """모델 출력에서 class token 추출, 실패 시 text2class 로 폴백."""
    # ── [class]N[/class] 형식 ──
    if "[class]" in pred and "[/class]" in pred:
        try:
            return int(pred.split("[/class]")[0].split("[class]")[-1])
        except Exception:
            pass
    # ── text2class 폴백 ──
    normed = normalize_label(pred)
    if normed in text2class:
        return text2class[normed]
    return 99


def parse_pred_text(pred: str) -> str:
    """모델 출력에서 text label 추출, 실패 시 전체 텍스트 반환."""
    if "[/class] [text]" in pred:
        try:
            return pred.split("[/class] [text]")[1].split("[/text")[0]
        except Exception:
            pass
    # [class] 없이 바로 텍스트만 생성된 경우
    return pred


# ─────────────────────────────────────────────────────────────────────────────
# 추론
# ─────────────────────────────────────────────────────────────────────────────

def generate_predictions(model, tokenizer, inputs, batch_size=32,
                         max_input_length=512, max_target_length=128, device="cpu"):
    """배치 추론으로 모델 출력 텍스트를 반환."""
    model.eval()
    all_preds = []
    for i in tqdm(range(0, len(inputs), batch_size), desc="  추론", leave=False):
        batch = inputs[i: i + batch_size]
        enc = tokenizer(
            batch,
            max_length=max_input_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out_ids = model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                max_new_tokens=max_target_length,
                num_beams=1,
                do_sample=False,
            )
        decoded = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
        all_preds.extend(decoded)
    return all_preds


# ─────────────────────────────────────────────────────────────────────────────
# 지표 계산
# ─────────────────────────────────────────────────────────────────────────────

def compute_freeform_metrics(gold_classes, gold_texts, preds):
    """
    Free-form 태스크 3개 지표 계산.
    반환: (C3, C2, TA) — 0~100 float
    """
    pred_classes = [parse_pred_class(p) for p in preds]
    pred_texts   = [parse_pred_text(p) for p in preds]

    # C(3): exact 3-way accuracy (-1 / 0 / 1)
    c3 = get_accuracy(gold_classes, pred_classes, accuracy_type="exact") * 100

    # C(2): binary accuracy (pos+disc vs neg)
    c2 = get_accuracy(gold_classes, pred_classes, accuracy_type="binary") * 100

    # T(A): 텍스트 출력 → text2class 변환 후 binary accuracy
    text_target_classes, text_pred_classes = \
        convert_moral_acceptability_text_to_class(gold_texts, pred_texts)
    ta = get_accuracy(text_target_classes, text_pred_classes, accuracy_type="binary") * 100

    return c3, c2, ta


def compute_yesno_metrics(gold_classes, gold_texts, preds):
    """
    Yes/no 태스크 2개 지표 계산.
    반환: (C2, TA) — 0~100 float
    """
    pred_classes = [parse_pred_class(p) for p in preds]
    pred_texts   = [parse_pred_text(p) for p in preds]

    # C(2): binary accuracy
    c2 = get_accuracy(gold_classes, pred_classes, accuracy_type="binary") * 100

    # T(A): polarity alignment accuracy
    _, ta = get_moral_agreement_text_accuracy(gold_texts, pred_texts)
    ta = ta * 100

    return c2, ta


# ─────────────────────────────────────────────────────────────────────────────
# 출력 포맷
# ─────────────────────────────────────────────────────────────────────────────

def print_checkpoint_result(step, metrics):
    """체크포인트별 상세 결과 테이블 출력."""
    line = "=" * 68
    print(f"\n{line}")
    print(f"  Checkpoint step: {step}")
    print(f"{line}")
    print(f"  {'지표':<22}  {'측정값':>8}  {'논문 목표':>9}  {'차이(Δ)':>9}")
    print(f"  {'-'*64}")
    for key, label in METRIC_LABELS:
        measured = metrics[key]
        target   = PAPER_TARGETS[key]
        delta    = measured - target
        sign     = "+" if delta >= 0 else ""
        print(f"  {label:<22}  {measured:>7.1f}%  {target:>8.1f}%  "
              f"{sign}{delta:>7.1f}%")
    print(line)


def print_summary(all_results):
    """전체 체크포인트 요약표 출력."""
    keys   = [k for k, _ in METRIC_LABELS]
    labels = ["C3-ff", "C2-ff", "TA-ff", "C2-yn", "TA-yn"]

    line = "=" * 72
    print(f"\n{line}")
    print("  전체 체크포인트 요약")
    print(f"{line}")
    header = f"  {'Step':>8}  " + "  ".join(f"{l:>7}" for l in labels)
    print(header)
    print(f"  {'-'*68}")
    for step, m in all_results:
        row = f"  {step:>8}  " + "  ".join(f"{m[k]:>6.1f}%" for k in keys)
        print(row)
    print(f"  {'-'*68}")
    paper_row = f"  {'(논문)':>8}  " + "  ".join(
        f"{PAPER_TARGETS[k]:>6.1f}%" for k in keys)
    print(paper_row)
    print(line)

    # 최고 체크포인트 추천 (C3 기준)
    if all_results:
        best_step, best_m = max(all_results, key=lambda x: x[1]["freeform_C(3)"])
        print(f"\n  ★ Free-form C(3) 기준 최고 체크포인트: step {best_step}"
              f"  ({best_m['freeform_C(3)']:.1f}%)")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────

@click.command()
@click.argument("results_dir", type=str)
@click.option("--pretrained-model",      default="large",         show_default=True,
              help="fine-tune.py 에 넘긴 pretrained_model 값")
@click.option("--mixture",               default="declare_only",  show_default=True,
              help="fine-tune.py 에 넘긴 mixture 값")
@click.option("--learning-rate",         default=2e-4,            show_default=True,
              help="fine-tune.py 에 넘긴 learning_rate 값")
@click.option("--batch-size",            default=16,              show_default=True,
              help="fine-tune.py 에 넘긴 batch_size 값")
@click.option("--checkpoint",            default=None, type=int,
              help="평가할 체크포인트 step (미지정 시 전체 평가)")
@click.option("--data-split",            default="validation",    show_default=True,
              help="평가 데이터 스플릿 (validation / test)")
@click.option("--inference-batch-size",  default=32,              show_default=True,
              help="추론 배치 크기 (GPU VRAM에 맞게 조정)")
@click.option("--input-prefix",          default="",              show_default=True,
              help="모델 입력에 붙일 접두사 (예: '[moral_single]: ')")
@click.option("--freeform-tsv",          default=None,
              help="Freeform 검증 TSV 경로 (미지정 시 기본값 사용)")
@click.option("--yesno-tsv",             default=None,
              help="Yes/no 검증 TSV 경로 (미지정 시 기본값 사용)")
def evaluate_all(
    results_dir, pretrained_model, mixture, learning_rate, batch_size,
    checkpoint, data_split, inference_batch_size, input_prefix,
    freeform_tsv, yesno_tsv,
):
    """
    fine-tune.py 체크포인트를 로드하여 논문 Table 3 의 5가지 지표를 계산하고
    목표치와 비교한다.

    RESULTS_DIR : fine-tune.py 에 전달한 results_dir 인수와 동일한 경로
    """

    # ── 체크포인트 경로 ──────────────────────────────────────────────────────
    result_path = os.path.join(
        results_dir, pretrained_model, mixture,
        f"lr-{learning_rate}_bs-{batch_size}"
    )
    print(f"\n체크포인트 탐색: {result_path}")

    all_ckpts = find_checkpoints(result_path)
    if not all_ckpts:
        print(f"\n[ERROR] '{result_path}' 에서 checkpoint-N 디렉토리를 찾을 수 없습니다.")
        print("  fine-tune.py 실행 후 생성된 checkpoint-N 디렉토리가 있는지 확인하세요.")
        return

    if checkpoint is not None:
        selected = [(s, p) for s, p in all_ckpts if s == checkpoint]
        if not selected:
            avail = [s for s, _ in all_ckpts]
            print(f"\n[ERROR] checkpoint={checkpoint} 없음. 이용 가능: {avail}")
            return
    else:
        selected = all_ckpts

    print(f"평가 대상 체크포인트: {[s for s, _ in selected]}")

    # ── 검증 데이터 로드 ─────────────────────────────────────────────────────
    _ff_tsv = freeform_tsv or os.path.join(
        PROJECT_ROOT, "data", "v11_declare_only", "freeform", f"{data_split}.tsv")
    _yn_tsv = yesno_tsv or os.path.join(
        PROJECT_ROOT, "data", "v11_declare_only", "yesno", f"{data_split}.tsv")

    for path in [_ff_tsv, _yn_tsv]:
        if not os.path.exists(path):
            print(f"\n[ERROR] 데이터 파일 없음: {path}")
            return

    ff_inputs, ff_classes, ff_texts = load_validation_tsv(_ff_tsv)
    yn_inputs, yn_classes, yn_texts = load_validation_tsv(_yn_tsv)

    print(f"Freeform {data_split}: {len(ff_inputs):,} examples  |  {_ff_tsv}")
    print(f"Yes/no   {data_split}: {len(yn_inputs):,} examples  |  {_yn_tsv}")

    # 접두사 적용
    if input_prefix:
        ff_inputs = [input_prefix + s for s in ff_inputs]
        yn_inputs = [input_prefix + s for s in yn_inputs]
        print(f"입력 접두사 적용: '{input_prefix}'")

    # ── 디바이스 ─────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # ── 체크포인트별 평가 루프 ───────────────────────────────────────────────
    all_results = []

    for step, ckpt_path in selected:
        print(f"{'#'*60}")
        print(f"  checkpoint-{step}")
        print(f"  {ckpt_path}")
        print(f"{'#'*60}")

        # 모델 / 토크나이저 로드
        print("  모델 로드 중...")
        tokenizer = T5Tokenizer.from_pretrained(ckpt_path)
        model     = T5ForConditionalGeneration.from_pretrained(ckpt_path)
        model.to(device)

        # ── Freeform 추론 ────────────────────────────────────────────────
        print(f"  [1/2] Freeform 추론 ({len(ff_inputs):,} examples)...")
        ff_preds = generate_predictions(
            model, tokenizer, ff_inputs,
            batch_size=inference_batch_size, device=device)

        c3, c2_ff, ta_ff = compute_freeform_metrics(ff_classes, ff_texts, ff_preds)

        # ── Yes/no 추론 ──────────────────────────────────────────────────
        print(f"  [2/2] Yes/no 추론 ({len(yn_inputs):,} examples)...")
        yn_preds = generate_predictions(
            model, tokenizer, yn_inputs,
            batch_size=inference_batch_size, device=device)

        c2_yn, ta_yn = compute_yesno_metrics(yn_classes, yn_texts, yn_preds)

        metrics = {
            "freeform_C(3)": c3,
            "freeform_C(2)": c2_ff,
            "freeform_T(A)": ta_ff,
            "yesno_C(2)":    c2_yn,
            "yesno_T(A)":    ta_yn,
        }

        print_checkpoint_result(step, metrics)
        all_results.append((step, metrics))

        # 메모리 정리
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── 전체 요약 ────────────────────────────────────────────────────────────
    if len(all_results) > 1:
        print_summary(all_results)
    elif len(all_results) == 1:
        # 단일 체크포인트 요약 라인
        step, m = all_results[0]
        print(f"\n  논문과의 평균 차이: "
              f"{sum(abs(m[k] - PAPER_TARGETS[k]) for k, _ in METRIC_LABELS) / len(METRIC_LABELS):.2f}%p\n")


if __name__ == "__main__":
    evaluate_all()
