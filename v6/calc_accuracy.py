#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute accuracy for your batch outputs.

Reads jsonl prediction files (each line should include:
  - model_answer
  - gold_final_answer (or gold_final_answer can be missing; those rows are skipped)

Reports:
  - strict exact-match accuracy
  - normalized exact-match accuracy (more forgiving: trim, lowercase, normalize punctuation/whitespace)

Also writes an error analysis file with mismatches.

One-click default (if files exist):
  test/out_primary_15_predictions.jsonl
  test/out_middle_15_predictions.jsonl
  test/out_high_15_predictions.jsonl

Usage:
  python calc_accuracy.py
  python calc_accuracy.py --inputs test/out_primary_15_predictions.jsonl test/out_middle_15_predictions.jsonl
  python calc_accuracy.py --out_csv test/acc_mismatches.csv
"""

import argparse
import csv
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, Any, List, Tuple


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def normalize_text(s: str) -> str:
    """
    A light normalization:
      - Unicode NFKC (turn fullwidth to halfwidth)
      - strip spaces/newlines
      - lowercase
      - unify common punctuation variants
      - collapse whitespace
    """
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKC", s)
    s = s.strip().lower()

    # unify punctuation
    s = s.replace("：", ":").replace("，", ",").replace("。", ".").replace("（", "(").replace("）", ")")

    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()

    return s


def score_file(path: Path) -> Tuple[int, int, int, int, List[Dict[str, Any]]]:
    """
    Returns:
      total_scored, strict_correct, norm_correct, skipped, mismatches(list)
    """
    rows = read_jsonl(path)
    total_scored = 0
    strict_correct = 0
    norm_correct = 0
    skipped = 0
    mismatches: List[Dict[str, Any]] = []

    for r in rows:
        gold = r.get("gold_final_answer")
        pred = r.get("model_answer")

        if gold is None:
            skipped += 1
            continue

        total_scored += 1

        gold_s = str(gold).strip()
        pred_s = "" if pred is None else str(pred).strip()

        if pred_s == gold_s:
            strict_correct += 1
        if normalize_text(pred_s) == normalize_text(gold_s):
            norm_correct += 1

        if pred_s != gold_s:
            mismatches.append({
                "file": path.name,
                "id": r.get("id"),
                "level": r.get("level"),
                "question": r.get("question"),
                "gold_final_answer": gold_s,
                "model_answer": pred_s,
                "model_answer_full": r.get("model_answer_full", ""),
            })

    return total_scored, strict_correct, norm_correct, skipped, mismatches


def pct(a: int, b: int) -> str:
    if b == 0:
        return "N/A"
    return f"{(a / b) * 100:.2f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="*", help="Prediction jsonl files to score")
    ap.add_argument("--out_csv", default=None, help="Write mismatches to CSV (default: test/acc_mismatches.csv if test/ exists)")
    args = ap.parse_args()

    base = Path(__file__).resolve().parent
    default_inputs = [
        base / "test" / "out_primary_15_predictions.jsonl",
        base / "test" / "out_middle_15_predictions.jsonl",
        base / "test" / "out_high_15_predictions.jsonl",
    ]

    inputs = [Path(p).expanduser().resolve() for p in (args.inputs or [])]
    if not inputs:
        inputs = [p for p in default_inputs if p.exists()]

    if not inputs:
        print("❌ No input files found. Use --inputs to specify prediction jsonl files.")
        return

    out_csv = args.out_csv
    if out_csv is None:
        test_dir = base / "test"
        out_csv = str((test_dir / "acc_mismatches.csv") if test_dir.exists() else (base / "acc_mismatches.csv"))

    grand_total = grand_strict = grand_norm = grand_skipped = 0
    all_mismatches: List[Dict[str, Any]] = []

    print("=== Accuracy Report ===")
    for p in inputs:
        if not p.exists():
            print(f"- {p}: ❌ missing")
            continue
        total_scored, strict_correct, norm_correct, skipped, mismatches = score_file(p)
        grand_total += total_scored
        grand_strict += strict_correct
        grand_norm += norm_correct
        grand_skipped += skipped
        all_mismatches.extend(mismatches)

        print(f"\nFile: {p}")
        print(f"  Scored: {total_scored} | Skipped(no gold): {skipped}")
        print(f"  Strict EM: {strict_correct}/{total_scored} = {pct(strict_correct, total_scored)}")
        print(f"  Norm   EM: {norm_correct}/{total_scored} = {pct(norm_correct, total_scored)}")

    print("\n--- Overall ---")
    print(f"Scored: {grand_total} | Skipped(no gold): {grand_skipped}")
    print(f"Strict EM: {grand_strict}/{grand_total} = {pct(grand_strict, grand_total)}")
    print(f"Norm   EM: {grand_norm}/{grand_total} = {pct(grand_norm, grand_total)}")

    # Write mismatches CSV
    out_csv_path = Path(out_csv).expanduser().resolve()
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with out_csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["file", "id", "level", "question", "gold_final_answer", "model_answer", "model_answer_full"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_mismatches:
            w.writerow(r)

    print(f"\nSaved mismatches CSV: {out_csv_path}")
    print(f"Mismatches: {len(all_mismatches)}")


if __name__ == "__main__":
    main()