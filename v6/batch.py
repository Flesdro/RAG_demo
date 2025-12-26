#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
import sys
import subprocess
from pathlib import Path
from typing import Dict, Any, List

# ========== 默认配置 ==========
DEFAULT_TIMEOUT_SEC = 180  # 更稳可改 300


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_answer_block(stdout: str) -> str:
    """
    Prefer: after the last Q> ... first A> block (ignore startup greetings).
    Fallback: last A> block.
    """
    # 1) last Q>
    q_idx = stdout.rfind("\nQ>")
    if q_idx == -1:
        q_idx = stdout.rfind("Q>")
    if q_idx != -1:
        tail = stdout[q_idx:]
        m = re.search(r"\nA>\s*", tail)
        if not m:
            m = re.search(r"^A>\s*", tail, flags=re.M)
        if m:
            ans = tail[m.end():]
            next_q = ans.find("\nQ>")
            if next_q != -1:
                ans = ans[:next_q]
            return ans.strip()

    # 2) fallback: last A>
    ms = list(re.finditer(r"\nA>\s*", stdout))
    if ms:
        m = ms[-1]
        ans = stdout[m.end():]
        next_q = ans.find("\nQ>")
        if next_q != -1:
            ans = ans[:next_q]
        return ans.strip()

    return stdout.strip()


def extract_final_answer_text(answer_block: str) -> str:
    """
    Return ONLY the content after the last '答案：' (or '最终答案：').
    Fallback: last non-empty line.
    """
    last = None
    for raw in answer_block.splitlines():
        line = re.sub(r"^\s*A>\s*", "", raw).strip()
        if not line:
            continue
        m = re.match(r"^(最终答案|答案)\s*[:：]\s*(.+?)\s*$", line)
        if m:
            last = m.group(2).strip()

    if last is not None:
        return last

    non_empty = [re.sub(r"^\s*A>\s*", "", ln).strip() for ln in answer_block.splitlines() if ln.strip()]
    return non_empty[-1] if non_empty else ""


def run_one(rag_py: Path, question: str, python_exe: str, timeout_sec: int) -> Dict[str, str]:
    """
    Returns dict:
      - model_answer (short)
      - model_answer_full (answer block; may be "")
      - status: OK / TIMEOUT / ERROR
    """
    inp = f"{question}\n/exit\n"
    try:
        proc = subprocess.run(
            [python_exe, str(rag_py)],
            input=inp,
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="ignore",
            timeout=timeout_sec,
        )
        out = proc.stdout if proc.stdout.strip() else (proc.stdout + "\n" + proc.stderr)
        block = extract_answer_block(out)
        final = extract_final_answer_text(block)
        return {
            "model_answer": final if final else "【EMPTY】",
            "model_answer_full": block,
            "status": "OK",
        }

    except subprocess.TimeoutExpired:
        return {"model_answer": "【TIMEOUT】", "model_answer_full": "", "status": "TIMEOUT"}
    except Exception as e:
        return {"model_answer": f"【ERROR】{type(e).__name__}: {e}", "model_answer_full": "", "status": "ERROR"}


def run_dataset(
    *,
    rag_py: Path,
    input_path: Path,
    out_path: Path,
    python_exe: str,
    timeout_sec: int,
    keep_full: bool,
) -> None:
    rows = read_jsonl(input_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f_out:
        for i, r in enumerate(rows, 1):
            q = r.get("question", "")
            res = run_one(rag_py, q, python_exe=python_exe, timeout_sec=timeout_sec)

            out_row: Dict[str, Any] = {
                "id": r.get("id"),
                "level": r.get("level"),
                "question": q,
                "model_answer": res["model_answer"],  # ✅ always short
                "status": res["status"],
            }
            if keep_full:
                out_row["model_answer_full"] = res["model_answer_full"]

            # 兼容两种 gold 字段名
            if "final_answer" in r:
                out_row["gold_final_answer"] = r["final_answer"]
            if "gold_final_answer" in r:
                out_row["gold_final_answer"] = r["gold_final_answer"]

            f_out.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            f_out.flush()

            print(f"[{input_path.name}] {i:02d}/{len(rows)}  {r.get('id')}  {res['status']}")

    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep_full", action="store_true", help="Also save model_answer_full (full answer block)")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SEC, help="Timeout seconds per question")
    ap.add_argument("--python", default=None, help="Python executable path (default: current interpreter)")
    args = ap.parse_args()

    base = Path(__file__).resolve().parent
    rag_py = base / "rag_v6_step_tree_intent_tone.py"
    test_dir = base / "test"

    python_exe = args.python or sys.executable

    inputs = [
        test_dir / "primary_15.jsonl",
        test_dir / "middle_15.jsonl",
        test_dir / "high_15.jsonl",
    ]
    outputs = [
        test_dir / ("out_primary_15_short_full.jsonl" if args.keep_full else "out_primary_15_short.jsonl"),
        test_dir / ("out_middle_15_short_full.jsonl" if args.keep_full else "out_middle_15_short.jsonl"),
        test_dir / ("out_high_15_short_full.jsonl" if args.keep_full else "out_high_15_short.jsonl"),
    ]

    missing = [p for p in [rag_py, *inputs] if not p.exists()]
    if missing:
        print("❌ Missing files:")
        for p in missing:
            print(" -", p)
        print("\nTip: put this script in the same folder as rag_v6_step_tree_intent_tone.py, and ensure test/*.jsonl exist.")
        return

    for inp, outp in zip(inputs, outputs):
        run_dataset(
            rag_py=rag_py,
            input_path=inp,
            out_path=outp,
            python_exe=python_exe,
            timeout_sec=args.timeout,
            keep_full=args.keep_full,
        )

    print("\n✅ All done.")


if __name__ == "__main__":
    main()