#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ollama pure-LLM baselines (Qwen2.5 etc.) with two prompt modes:
  - naive   : minimal prompt, shows "natural" hallucination tendency
  - guarded : strict "no guessing" prompt, forces refusal on insufficient info
  - both    : run both modes and write two output files per input

Default inputs (if no --inputs):
  test/primary_15.jsonl
  test/middle_15.jsonl
  test/high_15.jsonl
  test/hallucination_triggers.jsonl (if exists)

Outputs (default out_dir is ./test):
  out_<stem>_ollama_naive.jsonl
  out_<stem>_ollama_guarded.jsonl

You can redirect all outputs to a folder:
  --out_dir LLM_test

Usage:
  python ollama_baseline_batch_v2.py --model qwen2.5 --mode both --keep_full --out_dir LLM_test
  python ollama_baseline_batch_v2.py --mode guarded --inputs test/hallucination_triggers.jsonl --out_dir LLM_test
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional


DEFAULT_MODEL = "qwen2.5"
DEFAULT_TIMEOUT_SEC = 120

PROMPT_NAIVE = """你是一个数学解题助手。请直接回答问题。
题目：{question}
"""

PROMPT_GUARDED = """你是一个严谨的数学解题助手。
规则（非常重要）：
1) 只能根据题目给出的信息作答，绝对不要自行“假设/设/令/不妨/随便取”任何数值。
2) 如果题目缺少必要条件，或需要图片但未提供图片，或答案不唯一，请只输出：资料中没有找到
3) 如果可以算出唯一结果，请只输出最终答案（不要步骤，不要解释，不要多余文字）。
4) 输出中不要出现“答案：”前缀，只输出答案本身或“资料中没有找到”。

题目：{question}
"""


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def postprocess_answer(text: str) -> str:
    """
    Compact & comparable answer:
      - strip
      - take last non-empty line
      - remove "答案：" / "最终答案：" prefix if any slipped through
    """
    t = (text or "").strip()
    if not t:
        return "【EMPTY】"
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    t = lines[-1] if lines else t
    t = re.sub(r"^(最终答案|答案)\s*[:：]\s*", "", t).strip()
    return t if t else "【EMPTY】"


def ollama_run(model: str, prompt: str, timeout_sec: int) -> Dict[str, str]:
    """
    Calls: ollama run <model> <prompt>
    Returns dict: {status: OK/TIMEOUT/ERROR, full: raw_text_or_error}
    """
    try:
        proc = subprocess.run(
            ["ollama", "run", model, prompt],
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="ignore",
            timeout=timeout_sec,
        )
        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        full = out if out else err

        if proc.returncode != 0 and not full:
            return {"status": "ERROR", "full": f"ollama returncode={proc.returncode}"}

        return {"status": "OK", "full": full}

    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT", "full": ""}
    except FileNotFoundError:
        return {"status": "ERROR", "full": "ollama command not found. Please install Ollama and ensure it's in PATH."}
    except Exception as e:
        return {"status": "ERROR", "full": f"{type(e).__name__}: {e}"}


def default_inputs(script_dir: Path) -> List[Path]:
    test_dir = script_dir / "test"
    candidates = [
        test_dir / "primary_15.jsonl",
        test_dir / "middle_15.jsonl",
        test_dir / "high_15.jsonl",
        test_dir / "hallucination_triggers.jsonl",
    ]
    return [p for p in candidates if p.exists()]


def build_out_path(out_dir: Path, input_path: Path, mode: str) -> Path:
    # input stem like "primary_15" or "hallucination_triggers"
    return out_dir / f"out_{input_path.stem}_ollama_{mode}.jsonl"


def pick_prompt(mode: str) -> str:
    if mode == "naive":
        return PROMPT_NAIVE
    if mode == "guarded":
        return PROMPT_GUARDED
    raise ValueError(f"unknown mode: {mode}")


def run_dataset(
    input_path: Path,
    out_path: Path,
    model: str,
    mode: str,
    timeout_sec: int,
    keep_full: bool,
) -> None:
    rows = read_jsonl(input_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    template = pick_prompt(mode)

    with out_path.open("w", encoding="utf-8") as f_out:
        for i, r in enumerate(rows, 1):
            q = r.get("question", "") or ""
            prompt = template.format(question=q)

            res = ollama_run(model=model, prompt=prompt, timeout_sec=timeout_sec)
            if res["status"] == "OK":
                ans = postprocess_answer(res["full"])
            elif res["status"] == "TIMEOUT":
                ans = "【TIMEOUT】"
            else:
                ans = f"【ERROR】{res['full']}"

            out_row: Dict[str, Any] = {
                "id": r.get("id"),
                "level": r.get("level"),
                "question": q,
                "model": model,
                "baseline_mode": mode,
                "model_answer": ans,
                "status": res["status"],
            }

            # keep metadata (useful for hallucination eval & accuracy)
            for k in ["category", "expected", "gold_final_answer", "final_answer", "notes"]:
                if k in r:
                    out_row[k] = r[k]
            # unify gold
            if "final_answer" in out_row and "gold_final_answer" not in out_row:
                out_row["gold_final_answer"] = out_row["final_answer"]

            if keep_full:
                out_row["model_answer_full"] = res["full"]

            f_out.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            f_out.flush()

            print(f"[{input_path.name}][{mode}] {i:02d}/{len(rows)}  {r.get('id')}  {res['status']}")

    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model name (default: {DEFAULT_MODEL})")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SEC, help="Timeout seconds per question")
    ap.add_argument("--keep_full", action="store_true", help="Save model_answer_full (raw output)")
    ap.add_argument("--mode", choices=["naive", "guarded", "both"], default="both", help="Baseline prompt mode")
    ap.add_argument("--inputs", nargs="*", help="Input jsonl files (if omitted, uses default test/*.jsonl)")
    ap.add_argument("--out_dir", default=None, help="Directory to store all outputs (default: ./test if exists else .)")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent

    inputs = [Path(p).expanduser().resolve() for p in (args.inputs or [])]
    if not inputs:
        inputs = default_inputs(script_dir)

    if not inputs:
        print("❌ No input files found. Put datasets under ./test or pass --inputs ...")
        sys.exit(2)

    # choose output dir
    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        out_dir = (script_dir / "test") if (script_dir / "test").exists() else script_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    # light sanity check
    try:
        subprocess.run(["ollama", "--version"], capture_output=True, text=True, timeout=5)
    except Exception:
        pass

    modes = ["naive", "guarded"] if args.mode == "both" else [args.mode]

    for inp in inputs:
        if not inp.exists():
            print(f"❌ Missing input: {inp}")
            continue
        for m in modes:
            out_path = build_out_path(out_dir, inp, m)
            run_dataset(
                input_path=inp,
                out_path=out_path,
                model=args.model,
                mode=m,
                timeout_sec=args.timeout,
                keep_full=args.keep_full,
            )

    print("\n✅ All done.")


if __name__ == "__main__":
    main()