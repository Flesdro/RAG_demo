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

# 解析时用到的关键词（尽量兼容你不同版本的输出）
ANSWER_PREFIX_RE = re.compile(r"^(最终答案|答案)\s*[:：]\s*(.+?)\s*$")
A_PROMPT_RE = re.compile(r"(^|\n)A>\s*")
Q_PROMPT_RE = re.compile(r"(^|\n)Q>\s*")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def extract_answer_block(stdout: str) -> str:
    """
    目标：拿到“本题”的回答块，尽量避开启动开场白。

    策略：
    1) 优先取最后一次出现的 Q> 之后的第一个 A> 块
    2) 否则取最后一次出现的 A> 块
    3) 再不行就返回全文
    """
    s = stdout or ""

    # 1) last Q>
    q_idx = s.rfind("\nQ>")
    if q_idx == -1:
        q_idx = s.rfind("Q>")
    if q_idx != -1:
        tail = s[q_idx:]
        m = re.search(r"\nA>\s*", tail)
        if not m:
            m = re.search(r"^A>\s*", tail, flags=re.M)
        if m:
            ans = tail[m.end():]
            # cut at next Q> if any
            next_q = ans.find("\nQ>")
            if next_q != -1:
                ans = ans[:next_q]
            return ans.strip()

    # 2) last A>
    ms = list(re.finditer(r"\nA>\s*", s))
    if ms:
        m = ms[-1]
        ans = s[m.end():]
        next_q = ans.find("\nQ>")
        if next_q != -1:
            ans = ans[:next_q]
        return ans.strip()

    return s.strip()


def extract_final_answer_text(answer_block: str) -> str:
    """
    提取“只要最后答案”的字符串。
    - 优先找最后一条形如：答案：xxx / 最终答案：xxx
    - 否则：取最后一个非空行
    """
    last = None
    for raw in (answer_block or "").splitlines():
        line = re.sub(r"^\s*A>\s*", "", raw).strip()
        if not line:
            continue
        m = ANSWER_PREFIX_RE.match(line)
        if m:
            last = m.group(2).strip()

    if last is not None:
        return last if last else "【EMPTY】"

    lines = [re.sub(r"^\s*A>\s*", "", ln).strip() for ln in (answer_block or "").splitlines() if ln.strip()]
    return lines[-1] if lines else "【EMPTY】"


def run_one(rag_py: Path, question: str, python_exe: str, timeout_sec: int) -> Dict[str, str]:
    """
    用子进程运行 rag_v2.1.py，喂入：question + /exit
    返回：status, model_answer(short), model_answer_full(block)
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
        out = proc.stdout if (proc.stdout or "").strip() else ((proc.stdout or "") + "\n" + (proc.stderr or ""))
        block = extract_answer_block(out)
        short = extract_final_answer_text(block)
        return {"status": "OK", "model_answer": short, "model_answer_full": block}

    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT", "model_answer": "【TIMEOUT】", "model_answer_full": ""}

    except Exception as e:
        return {"status": "ERROR", "model_answer": f"【ERROR】{type(e).__name__}: {e}", "model_answer_full": ""}


def default_inputs(script_dir: Path) -> List[Path]:
    test_dir = script_dir / "test"
    candidates = [
        test_dir / "primary_15.jsonl",
        test_dir / "middle_15.jsonl",
        test_dir / "high_15.jsonl",
        test_dir / "hallucination_triggers.jsonl",
    ]
    return [p for p in candidates if p.exists()]


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
            q = r.get("question", "") or ""
            res = run_one(rag_py, q, python_exe=python_exe, timeout_sec=timeout_sec)

            out_row: Dict[str, Any] = {
                "id": r.get("id"),
                "level": r.get("level"),
                "question": q,
                "model": "rag_v2.1",
                "model_answer": res["model_answer"],  # ✅ 短答案
                "status": res["status"],
            }
            # 保留原数据的 metadata，方便你后续 hallucination_rate / accuracy 直接吃
            for k in ["category", "expected", "notes", "gold_final_answer", "final_answer"]:
                if k in r:
                    out_row[k] = r[k]
            if "final_answer" in out_row and "gold_final_answer" not in out_row:
                out_row["gold_final_answer"] = out_row["final_answer"]

            if keep_full:
                out_row["model_answer_full"] = res["model_answer_full"]

            f_out.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            f_out.flush()

            print(f"[{input_path.name}] {i:02d}/{len(rows)}  {r.get('id')}  {res['status']}")

    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rag", default=None, help="path to rag_v2.1.py (default: ./rag_v2.1.py)")
    ap.add_argument("--python", default=None, help="python executable (default: current interpreter)")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SEC, help="timeout per question (sec)")
    ap.add_argument("--keep_full", action="store_true", help="also save model_answer_full")
    ap.add_argument("--inputs", nargs="*", help="input jsonl files (default: ./test/*.jsonl)")
    ap.add_argument("--out_dir", default=None, help="output directory (e.g., LLM_test). default: ./test if exists else .")
    ap.add_argument("--tag", default="rag21", help="filename tag prefix for outputs")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    python_exe = args.python or sys.executable

    rag_py = Path(args.rag).expanduser().resolve() if args.rag else (script_dir / "rag_v2.1.py")
    if not rag_py.exists():
        print(f"❌ rag file not found: {rag_py}")
        return

    inputs = [Path(p).expanduser().resolve() for p in (args.inputs or [])]
    if not inputs:
        inputs = default_inputs(script_dir)

    if not inputs:
        print("❌ No input files found. Put datasets under ./test or pass --inputs ...")
        return

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else ((script_dir / "test") if (script_dir / "test").exists() else script_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for inp in inputs:
        if not inp.exists():
            print(f"❌ Missing input: {inp}")
            continue
        out_path = out_dir / f"{args.tag}_out_{inp.stem}.jsonl"
        run_dataset(
            rag_py=rag_py,
            input_path=inp,
            out_path=out_path,
            python_exe=python_exe,
            timeout_sec=args.timeout,
            keep_full=args.keep_full,
        )

    print("\n✅ All done.")


if __name__ == "__main__":
    main()