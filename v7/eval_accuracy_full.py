#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_accuracy_full.py

对齐 gold/pred 的 jsonl（按 id），计算准确率（Answer Match Accuracy）。
特点：
- 兼容 LaTeX 包裹：\( \), \frac{a}{b}, \sqrt{...}
- 兼容隐式乘法：4x -> 4*x（若安装 sympy）
- 兼容多解：x=2或x=3 / x=3,x=2 / x=±7
- 兼容不等式/区间：x>3 / 3<x / -1<x<4 / (-1,4) / x<=2 或 x>=3
- 兼容点集：(3,4)和(-4,-3)
- 兼容“带文字的答案”：例如“行驶了150千米。”、“最小值为3，当x=2时取到最小值。”
- 兼容时间：1小时45分钟 与 105分钟（会统一到总分钟数；注意：175分钟不是 1小时45分钟）

用法：
  python eval_accuracy_full.py --gold test/high_15.jsonl --pred test_result/pred_high_15.jsonl
  python eval_accuracy_full.py --gold test/middle_15.jsonl --pred test_result/pred_middle_15.jsonl
  python eval_accuracy_full.py --gold test/primary_15.jsonl --pred test_result/pred_primary_15.jsonl

可选：
  --show_mismatch 10            展示前 N 条不匹配样例
  --out_mismatch mism.jsonl     把不匹配样例写入 jsonl
  --by_mode                     额外按 used_mode 统计准确率
"""

import argparse
import json
import re
from typing import Dict, Any, List, Optional, Tuple, Set

# Sympy is optional but recommended for robust equivalence
try:
    import sympy as sp
    from sympy.parsing.sympy_parser import (
        parse_expr,
        standard_transformations,
        implicit_multiplication_application,
        convert_xor,
    )
    _TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)
except Exception:
    sp = None
    parse_expr = None
    _TRANSFORMS = None


# ---------------- I/O ----------------

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------- Normalization ----------------

def strip_latex(s: str) -> str:
    """去掉常见 LaTeX 包裹，并把 \\frac/\\sqrt 转成更可解析格式。"""
    if s is None:
        return ""
    s = str(s)

    # 去掉 \( \)  \[ \]  $$ $$
    s = s.replace("\\(", "").replace("\\)", "")
    s = s.replace("\\[", "").replace("\\]", "")
    s = s.replace("$$", "").replace("$", "")

    # \frac{a}{b} -> (a)/(b)（一层花括号即可覆盖你的数据）
    s = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", s)

    # \sqrt{2} -> sqrt(2)
    s = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", s)

    # 去掉剩余反斜杠（比如 \theta）
    s = s.replace("\\", "")
    return s


def normalize_text(s: str) -> str:
    """统一符号、空白、le/ge 等，尽量转成接近 sympy / 易比较的形式。"""
    s = strip_latex(s)
    s = str(s).strip()

    # 中文标点/符号
    s = s.replace("＝", "=").replace("×", "*").replace("÷", "/")
    s = s.replace("（", "(").replace("）", ")")
    s = s.replace("，", ",").replace("；", ";").replace("：", ":")
    s = s.replace("≤", "<=").replace("≥", ">=").replace("≠", "!=")

    # 兼容 gold 里的 le/ge（注意空格写法：x le 2）
    s = re.sub(r"\ble\b", "<=", s)
    s = re.sub(r"\bge\b", ">=", s)

    # 去掉“答案：”前缀
    s = re.sub(r"^\s*(答案|答)\s*[:：]\s*", "", s)

    # 统一空白
    s = re.sub(r"\s+", " ", s).strip()

    # 句尾标点
    s = s.strip("。.")

    # 幂符号
    s = s.replace("^", "**")

    return s


def split_outside_parens(s: str, seps: Set[str]) -> List[str]:
    """按分隔符切分，但仅在括号外切分（避免 (-1,4) / log(x,2) 被误切）。"""
    parts: List[str] = []
    buf: List[str] = []
    depth = 0
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")" and depth > 0:
            depth -= 1

        if depth == 0 and ch in seps:
            piece = "".join(buf).strip()
            if piece:
                parts.append(piece)
            buf = []
        else:
            buf.append(ch)

    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def split_multi(s: str) -> List[str]:
    """
    拆多解（或/和/and/逗号/分号等），但保留括号内逗号。
    """
    s = normalize_text(s)
    # 先把连接词替换成分号
    s = re.sub(r"\s*(或|和|以及|并且|and)\s*", ";", s, flags=re.I)

    parts: List[str] = []
    for chunk in s.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        # 再按逗号（括号外）拆
        parts.extend(split_outside_parens(chunk, seps={",", "，"}))

    return [p.strip() for p in parts if p.strip()]


# ---------------- Sympy parsing ----------------

def try_sympy(expr: str):
    """尽量把字符串解析成 sympy 表达式（支持隐式乘法与常见 LaTeX 残留）。"""
    if sp is None or parse_expr is None:
        return None

    raw = strip_latex(expr).strip()

    # e^{...} / e^x -> exp(...)
    raw = re.sub(r"e\^\{([^}]+)\}", r"exp(\1)", raw)
    raw = re.sub(r"e\^([A-Za-z0-9_]+)", r"exp(\1)", raw)

    # 角度符号：30° -> 30*pi/180
    raw = re.sub(r"(\d+(?:\.\d+)?)°", r"(\1*pi/180)", raw)

    # 基础规范
    t = raw
    t = t.replace("＝", "=").replace("×", "*").replace("÷", "/")
    t = t.replace("（", "(").replace("）", ")")
    t = t.replace("，", ",").replace("；", ";").replace("：", ":")
    t = t.replace("≤", "<=").replace("≥", ">=").replace("≠", "!=")
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"^\s*(答案|答)\s*[:：]\s*", "", t)
    t = t.strip("。.")
    t = re.sub(r"\ble\b", "<=", t)
    t = re.sub(r"\bge\b", ">=", t)
    t = t.replace("^", "**")
    t = t.replace("√", "sqrt").replace("π", "pi")

    # 兼容 e**{2x} 这类（避免 set 语法）
    t = re.sub(r"e\*\*\{([^}]+)\}", r"exp(\1)", t)

    try:
        return sp.simplify(parse_expr(t, transformations=_TRANSFORMS))
    except Exception:
        try:
            return sp.simplify(sp.sympify(t))
        except Exception:
            return None


def canonical_value(s: str) -> str:
    s = normalize_text(s)
    v = try_sympy(s)
    if v is not None:
        return str(v)
    return s.replace(" ", "")


def _clean_rhs(rhs: str) -> str:
    """
    清理赋值右侧，把“2时取到最小值”截成“2”；
    对于纯表达式（4x^3-6x+2）会保留完整。
    """
    rhs = normalize_text(rhs).strip()
    rhs_nospace = rhs.replace(" ", "")
    m = re.match(r"^[0-9A-Za-z_\.\+\-\*/\(\)]+", rhs_nospace)
    if m:
        return m.group(0)
    return rhs_nospace


# ---------------- Answer structure parsing ----------------

def parse_points(s: str) -> Optional[Set[Tuple[float, float]]]:
    t = normalize_text(s).replace(" ", "")
    pts = re.findall(r"\((-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)\)", t)
    if not pts:
        return None
    return {(float(a), float(b)) for a, b in pts}


def parse_interval(s: str) -> Optional[Tuple[str, float, float]]:
    t = normalize_text(s).replace(" ", "")
    m = re.fullmatch(r"\((-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)\)", t)
    if m:
        return ("open", float(m.group(1)), float(m.group(2)))
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)<x<(-?\d+(?:\.\d+)?)", t)
    if m:
        return ("open", float(m.group(1)), float(m.group(2)))
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)<=x<=(-?\d+(?:\.\d+)?)", t)
    if m:
        return ("closed", float(m.group(1)), float(m.group(2)))
    return None


def parse_half_ineq(s: str) -> Optional[Tuple[str, str, float]]:
    t = normalize_text(s).replace(" ", "")
    # 3<x  / 3<=x
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)(<|<=)x", t)
    if m:
        num = float(m.group(1))
        op = m.group(2)
        return ("x", ">" if op == "<" else ">=", num)
    # x>3 / x<=2
    m = re.fullmatch(r"x(>|>=|<|<=)(-?\d+(?:\.\d+)?)", t)
    if m:
        return ("x", m.group(1), float(m.group(2)))
    return None


def parse_inequalities(s: str) -> Optional[Set[Tuple]]:
    parts = split_multi(s)
    comps: List[Tuple] = []
    for p in parts:
        itv = parse_interval(p)
        if itv:
            comps.append(("interval",) + itv)
            continue
        half = parse_half_ineq(p)
        if half:
            comps.append(("half",) + half)
            continue
    if comps:
        return set(comps)
    return None


def parse_time_minutes(s: str) -> Optional[int]:
    t = normalize_text(s)
    h = re.search(r"(\d+)\s*小时", t)
    m = re.search(r"(\d+)\s*分钟", t)
    if h or m:
        hh = int(h.group(1)) if h else 0
        mm = int(m.group(1)) if m else 0
        return hh * 60 + mm

    m2 = re.fullmatch(r"(\d+)\s*分钟", t.replace(" ", ""))
    if m2:
        return int(m2.group(1))
    return None


def extract_assignments(s: str) -> List[Tuple[str, str]]:
    s = normalize_text(s)
    s = re.sub(r"\s*=\s*", "=", s)

    # x=±7 -> x=7 和 x=-7
    m = re.search(r"([A-Za-z][A-Za-z0-9_'\(\)]*)=±\s*([-\d\.]+)", s)
    if m:
        var = m.group(1)
        val = m.group(2)
        return [(var, val), (var, "-" + val)]

    # lhs=rhs, rhs 可包含空格，但不跨越逗号/分号
    pairs = re.findall(r"([A-Za-z][A-Za-z0-9_'\(\)]*)=([^,;，；]+)", s)

    out: List[Tuple[str, str]] = []
    for a, b in pairs:
        out.append((a.strip(), _clean_rhs(b)))
    return out


def numeric_values(s: str) -> Set[str]:
    """
    提取“可比较的值集合”：
    - sympy 化简后的整体表达式（如果能解析）
    - 所有数字（用于“行驶了150千米。”这种）
    """
    t = normalize_text(s)
    vals: Set[str] = set()

    v = try_sympy(t)
    if v is not None:
        vals.add(str(v))

    # 提取数字
    nums = re.findall(r"-?\d+(?:\.\d+)?", t)
    for n in nums:
        vals.add(n.lstrip("+"))

    # 分数 a/b 也会被 sympy 化成 Rational；若未成功则保留原形式
    if re.fullmatch(r"-?\d+\s*/\s*\d+", t.replace(" ", "")):
        vals.add(t.replace(" ", ""))

    return vals


def normalize_solution(s: str) -> Dict[str, Any]:
    s0 = normalize_text(s)
    out: Dict[str, Any] = {
        "points": None,
        "ineq": None,
        "assign": set(),   # set of (lhs, rhs_canon)
        "time": None,      # int minutes
        "values": set(),   # set of canonical numeric/expr strings
        "raw": s0,
    }

    # 不等式优先（用于 disambiguate "(a,b)"）
    ineq = parse_inequalities(s0)
    if ineq:
        out["ineq"] = ineq

    # 点：若是单个 (a,b) 且可视为区间，就不当点
    pts = parse_points(s0)
    if pts:
        if len(pts) == 1 and parse_interval(s0) is not None:
            pass
        else:
            out["points"] = pts

    # 时间
    tm = parse_time_minutes(s0)
    if tm is not None:
        out["time"] = tm

    # 赋值对（多解聚合）
    for part in split_multi(s0):
        for lhs, rhs in extract_assignments(part):
            out["assign"].add((lhs, canonical_value(rhs)))

    # 值集合
    out["values"] = numeric_values(s0)

    return out


def answers_match(pred: str, gold: str) -> bool:
    p = normalize_solution(pred)
    g = normalize_solution(gold)

    # 交点/坐标
    if g["points"] is not None or p["points"] is not None:
        return g["points"] == p["points"]

    # 不等式/区间
    if g["ineq"] is not None or p["ineq"] is not None:
        return g["ineq"] == p["ineq"]

    # 方程解/函数值点（只要 pred 至少包含 gold 的赋值对即可）
    if g["assign"]:
        return g["assign"].issubset(p["assign"])

    # 时间：优先比分钟
    if g["time"] is not None:
        if p["time"] is not None and p["time"] == g["time"]:
            return True
        # 允许 pred 直接给出“105分钟”
        return str(g["time"]) in p["values"]

    # 纯值：至少一个 canonical value 交集
    if g["values"]:
        return len(g["values"] & p["values"]) > 0

    # 兜底：纯文本一致
    return normalize_text(pred).replace(" ", "") == normalize_text(gold).replace(" ", "")


# ---------------- Evaluation ----------------

def evaluate(gold_rows: List[Dict[str, Any]],
             pred_rows: List[Dict[str, Any]],
             id_field: str,
             gold_field: str,
             pred_field: str,
             by_mode: bool = False):
    gold_map = {str(r.get(id_field)): r for r in gold_rows if r.get(id_field) is not None}
    pred_map = {str(r.get(id_field)): r for r in pred_rows if r.get(id_field) is not None}

    common_ids = sorted(set(gold_map) & set(pred_map))
    missing_in_pred = sorted(set(gold_map) - set(pred_map))
    extra_in_pred = sorted(set(pred_map) - set(gold_map))

    total = len(common_ids)
    correct = 0
    mismatches: List[Dict[str, Any]] = []

    # by_mode stats
    mode_stats: Dict[str, Dict[str, int]] = {}

    for _id in common_ids:
        g = gold_map[_id].get(gold_field, "")
        p = pred_map[_id].get(pred_field, "")
        mode = pred_map[_id].get("used_mode", "UNKNOWN")

        ok = answers_match(str(p), str(g))
        if ok:
            correct += 1

        if by_mode:
            mode_stats.setdefault(mode, {"total": 0, "correct": 0})
            mode_stats[mode]["total"] += 1
            if ok:
                mode_stats[mode]["correct"] += 1

        if not ok:
            mismatches.append({
                "id": _id,
                "question": gold_map[_id].get("question", ""),
                "gold": g,
                "pred": p,
                "used_mode": mode,
            })

    acc = correct / total if total else 0.0
    return {
        "total": total,
        "correct": correct,
        "accuracy": acc,
        "missing_in_pred": missing_in_pred,
        "extra_in_pred": extra_in_pred,
        "mismatches": mismatches,
        "mode_stats": mode_stats,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True, help="gold jsonl（含 id, question, final_answer）")
    ap.add_argument("--pred", required=True, help="pred jsonl（含 id, final_answer, used_mode 可选）")
    ap.add_argument("--id_field", default="id", help="默认 id")
    ap.add_argument("--gold_field", default="final_answer", help="gold 答案字段名")
    ap.add_argument("--pred_field", default="final_answer", help="pred 答案字段名")
    ap.add_argument("--show_mismatch", type=int, default=10, help="展示前 N 条不匹配样例")
    ap.add_argument("--out_mismatch", default="", help="把不匹配样例写入 jsonl（可选）")
    ap.add_argument("--by_mode", action="store_true", help="按 used_mode 统计准确率")
    args = ap.parse_args()

    gold_rows = read_jsonl(args.gold)
    pred_rows = read_jsonl(args.pred)

    report = evaluate(
        gold_rows=gold_rows,
        pred_rows=pred_rows,
        id_field=args.id_field,
        gold_field=args.gold_field,
        pred_field=args.pred_field,
        by_mode=args.by_mode,
    )

    print("=" * 80)
    print(f"Gold: {args.gold}")
    print(f"Pred: {args.pred}")
    print(f"Matched by id: {report['total']}")
    print(f"Correct: {report['correct']}")
    print(f"Accuracy: {report['accuracy']:.4f}")

    print("-" * 80)
    print(f"Missing in pred: {len(report['missing_in_pred'])}")
    if report["missing_in_pred"][:10]:
        print("  e.g.", report["missing_in_pred"][:10])
    print(f"Extra in pred: {len(report['extra_in_pred'])}")
    if report["extra_in_pred"][:10]:
        print("  e.g.", report["extra_in_pred"][:10])

    if args.by_mode and report["mode_stats"]:
        print("=" * 80)
        print("Accuracy by used_mode:")
        for mode, st in sorted(report["mode_stats"].items(), key=lambda x: (-x[1]["total"], x[0])):
            acc = st["correct"] / st["total"] if st["total"] else 0.0
            print(f"- {mode:10s}  {st['correct']:3d}/{st['total']:3d}  acc={acc:.4f}")

    mismatches = report["mismatches"]
    if mismatches and args.show_mismatch > 0:
        print("=" * 80)
        print(f"Top {min(args.show_mismatch, len(mismatches))} mismatches:")
        for m in mismatches[: args.show_mismatch]:
            print("-" * 80)
            print("id:", m["id"], "used_mode:", m.get("used_mode", ""))
            if m.get("question"):
                print("Q:", m["question"])
            print("GOLD:", m["gold"])
            print("PRED:", m["pred"])

    if args.out_mismatch:
        write_jsonl(args.out_mismatch, mismatches)
        print("\nSaved mismatches ->", args.out_mismatch)

    print("=" * 80)


if __name__ == "__main__":
    main()
