#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
import sys
import subprocess
from pathlib import Path
from typing import Dict, Any, List

DEFAULT_TIMEOUT_SEC = 180


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
    Extract the answer AFTER the last Q> prompt (ignore the startup greetings).
    """
    # Prefer: after last Q>
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

    # Fallback: last A>
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
    Returns:
      status: OK / TIMEOUT / ERROR
      model_answer: short answer
      model_answer_full: answer block (may be "")
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
            "status": "OK",
            "model_answer": final if final else "【EMPTY】",
            "model_answer_full": block,
        }

    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT", "model_answer": "【TIMEOUT】", "model_answer_full": ""}
    except Exception as e:
        return {"status": "ERROR", "model_answer": f"【ERROR】{type(e).__name__}: {e}", "model_answer_full": ""}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep_full", action="store_true", help="Also save model_answer_full (recommended for analysis)")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SEC, help="Timeout seconds per question")
    ap.add_argument("--python", default=None, help="Python executable path (default: current interpreter)")
    ap.add_argument("--rag", default=None, help="Path to rag_v6_step_tree_intent_tone.py (default: same folder)")
    ap.add_argument("--input", default=None, help="Dataset jsonl (default: test/hallucination_triggers.jsonl)")
    ap.add_argument("--out", default=None, help="Output jsonl (default: test/out_hallucination_triggers_predictions.jsonl)")
    args = ap.parse_args()

    base = Path(__file__).resolve().parent
    python_exe = args.python or sys.executable

    rag_py = Path(args.rag).expanduser().resolve() if args.rag else (base / "rag_v6_step_tree_intent_tone.py")
    input_path = Path(args.input).expanduser().resolve() if args.input else (base / "test" / "hallucination_triggers.jsonl")
    out_path = Path(args.out).expanduser().resolve() if args.out else (base / "test" / "out_hallucination_triggers_predictions.jsonl")

    missing = [p for p in [rag_py, input_path] if not p.exists()]
    if missing:
        print("❌ Missing files:")
        for p in missing:
            print(" -", p)
        print("\nTip: put this script in the same folder as rag_v6_step_tree_intent_tone.py, and ensure test/hallucination_triggers.jsonl exists.")
        return

    rows = read_jsonl(input_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f_out:
        for i, r in enumerate(rows, 1):
            q = r.get("question", "")
            res = run_one(rag_py, q, python_exe=python_exe, timeout_sec=args.timeout)

            out_row: Dict[str, Any] = {
                "id": r.get("id"),
                "level": r.get("level"),
                "category": r.get("category"),
                "expected": r.get("expected"),
                "question": q,
                "model_answer": res["model_answer"],
                "status": res["status"],
            }
            if args.keep_full:
                out_row["model_answer_full"] = res["model_answer_full"]

            # keep gold if present
            if "gold_final_answer" in r:
                out_row["gold_final_answer"] = r["gold_final_answer"]

            f_out.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            f_out.flush()

            print(f"[hallucination_triggers.jsonl] {i:02d}/{len(rows)}  {r.get('id')}  {res['status']}")

    print(f"\nSaved: {out_path}")
    print("✅ Done.")


if __name__ == "__main__":
    main()