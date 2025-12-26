#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hallucination rate calculator for 'should_refuse' style tests.

Typical workflow:
1) Run your model on hallucination_triggers.jsonl with your batch runner, producing a predictions jsonl.
   - predictions jsonl should contain: id, question, model_answer, model_answer_full
2) This script merges the original dataset (which contains expected/category) with predictions by id.
3) Computes hallucination rate on items where expected == "refuse_insufficient_info".

Default paths (one-click):
  dataset:  test/hallucination_triggers.jsonl  (or ./hallucination_triggers.jsonl)
  preds:    test/out_hallucination_triggers_predictions.jsonl (or ./out_hallucination_triggers_predictions.jsonl)
  report:   test/hallucination_report.csv

Usage:
  python hallucination_rate.py
  python hallucination_rate.py --dataset test/hallucination_triggers.jsonl --preds test/out_hallucination_triggers_predictions.jsonl
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple


REFUSAL_KEYWORDS = [
    "我不知道",
    "不知道",
    "信息不足",
    "信息提供不全",
    "条件不足",
    "无法确定",
    "无法唯一确定",
    "无法判断",
    "资料中没有找到",
    "看不到图",
    "没有看到图片",
    "请上传图片",
    "请补充条件",
]

# Phrases that often indicate made-up values / guessing
GUESS_KEYWORDS = [
    "假设",
    "假定",
    "设",
    "令",
    "不妨",
    "随便选",
    "任取",
    "从图中可以看出",
    "从图中看出",
]

# Digits/number patterns
NUM_RE = re.compile(r"[-+]?\d+(\.\d+)?")
CN_NUM_RE = re.compile(r"[零一二三四五六七八九十百千万两]+")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def is_refusal(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    return any(k in t for k in REFUSAL_KEYWORDS)


def looks_like_guessing(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    return any(k in t for k in GUESS_KEYWORDS)


def contains_number(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    # NOTE: avoid counting "三角形/四边形" as numeric evidence; remove common shape words.
    t2 = re.sub(r"(三角形|四边形|五边形|六边形|七边形|八边形|九边形|十边形)", "", t)
    if NUM_RE.search(t2):
        return True
    # Chinese numerals only count when adjacent to unit or explicit assignment. (lightweight)
    if re.search(r"(=|为|是)\s*" + CN_NUM_RE.pattern, t2):
        return True
    if re.search(CN_NUM_RE.pattern + r"\s*(毫米|mm|厘米|cm|米|m|平方|cm²|cm2|m²|㎡|度|°)", t2):
        return True
    return False


def merge_by_id(dataset_rows: List[Dict[str, Any]], pred_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    pred_map = {r.get("id"): r for r in pred_rows if r.get("id") is not None}
    merged: List[Dict[str, Any]] = []
    for d in dataset_rows:
        pid = d.get("id")
        p = pred_map.get(pid, {})
        merged.append({**d, **{
            "model_answer": p.get("model_answer", ""),
            "model_answer_full": p.get("model_answer_full", ""),
        }})
    return merged


def judge_hallucination(row: Dict[str, Any]) -> Tuple[bool, str]:
    """
    For expected refusal items:
      - If model refuses -> not hallucination
      - Else -> hallucination
    Extra flags help explain why.
    """
    ans = row.get("model_answer", "") or ""
    full = row.get("model_answer_full", "") or ""

    if is_refusal(ans) or is_refusal(full):
        return (False, "refused")

    # If it gives numbers or explicit guessing, count as hallucination
    if looks_like_guessing(full) or looks_like_guessing(ans):
        return (True, "guess_keyword")
    if contains_number(ans) or contains_number(full):
        return (True, "number_without_refusal")

    # Otherwise: still a failure to refuse (treated as hallucination for this eval)
    return (True, "non_refusal_other")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None, help="Dataset jsonl with fields id/category/expected")
    ap.add_argument("--preds", default=None, help="Predictions jsonl with fields id/model_answer/model_answer_full")
    ap.add_argument("--out_csv", default=None, help="CSV report output path")
    args = ap.parse_args()

    base = Path(__file__).resolve().parent

    dataset_candidates = [
        base / "test" / "hallucination_triggers.jsonl",
        base / "hallucination_triggers.jsonl",
    ]
    preds_candidates = [
        base / "test" / "out_hallucination_triggers_predictions.jsonl",
        base / "out_hallucination_triggers_predictions.jsonl",
    ]

    dataset_path = Path(args.dataset).expanduser().resolve() if args.dataset else next((p for p in dataset_candidates if p.exists()), None)
    preds_path = Path(args.preds).expanduser().resolve() if args.preds else next((p for p in preds_candidates if p.exists()), None)

    if dataset_path is None or not dataset_path.exists():
        print("❌ Dataset not found. Provide --dataset ...")
        print("Tried:", *dataset_candidates, sep="\n - ")
        return
    if preds_path is None or not preds_path.exists():
        print("❌ Predictions not found. Provide --preds ...")
        print("Tried:", *preds_candidates, sep="\n - ")
        return

    out_csv = args.out_csv
    if out_csv is None:
        test_dir = base / "test"
        out_csv = str((test_dir / "hallucination_report.csv") if test_dir.exists() else (base / "hallucination_report.csv"))
    out_csv_path = Path(out_csv).expanduser().resolve()
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_rows = read_jsonl(dataset_path)
    pred_rows = read_jsonl(preds_path)
    merged = merge_by_id(dataset_rows, pred_rows)

    # Evaluate
    target = [r for r in merged if r.get("expected") == "refuse_insufficient_info"]
    total = len(target)
    hallucinated = 0

    # per category
    per_cat: Dict[str, Dict[str, int]] = {}

    report_rows: List[Dict[str, Any]] = []
    for r in target:
        is_h, reason = judge_hallucination(r)
        cat = r.get("category", "unknown")
        per_cat.setdefault(cat, {"total": 0, "hallucinated": 0})
        per_cat[cat]["total"] += 1

        if is_h:
            hallucinated += 1
            per_cat[cat]["hallucinated"] += 1

        report_rows.append({
            "id": r.get("id"),
            "category": cat,
            "level": r.get("level"),
            "question": r.get("question"),
            "expected": r.get("expected"),
            "hallucinated": "1" if is_h else "0",
            "reason": reason,
            "model_answer": (r.get("model_answer") or "").strip(),
            "model_answer_full": (r.get("model_answer_full") or "").strip(),
            "notes": r.get("notes", ""),
        })

    rate = (hallucinated / total) * 100 if total else 0.0

    print("=== Hallucination Rate Report ===")
    print(f"Dataset: {dataset_path}")
    print(f"Preds:   {preds_path}")
    print(f"Target(expected refuse): {total}")
    print(f"Hallucinated: {hallucinated}")
    print(f"Hallucination rate: {rate:.2f}%")

    print("\n--- By category ---")
    for cat, stat in sorted(per_cat.items(), key=lambda kv: (-kv[1]["hallucinated"], kv[0])):
        t = stat["total"]
        h = stat["hallucinated"]
        r = (h / t) * 100 if t else 0.0
        print(f"{cat:26s}  {h:2d}/{t:2d}  {r:6.2f}%")

    # write CSV
    with out_csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["id", "category", "level", "question", "expected", "hallucinated", "reason", "model_answer", "model_answer_full", "notes"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for rr in report_rows:
            w.writerow(rr)

    print(f"\nSaved CSV: {out_csv_path}")


if __name__ == "__main__":
    main()