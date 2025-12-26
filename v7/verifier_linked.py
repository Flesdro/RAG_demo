from __future__ import annotations

"""
verifier.py
=========================
新增：推理引擎的“校验器”模块

目标：
- 给 Sympy 的求解结果做快速校验，降低“解析错/推错”的风险
- 提供安全的表达式解析（支持隐式乘法，如 2x -> 2*x）

说明：
- 这是一个“轻量校验器”，不是完整的形式化证明系统
- 对恒等式/化简等，可用 simplify + 随机代入做双保险
"""

from dataclasses import dataclass
import random
import re
from typing import Dict, Iterable, List, Optional, Tuple

import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)


_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)


def normalize_math_text(s: str) -> str:
    """把中文/全角符号、常见数学符号统一成 sympy 友好格式。"""
    s = s.strip()
    # 常见运算符
    s = s.replace("×", "*").replace("÷", "/").replace("^", "**")
    s = s.replace("（", "(").replace("）", ")")
    s = s.replace("，", ",").replace("：", ":").replace("；", ";")

    # 不等号/比较符
    s = s.replace("≤", "<=").replace("≥", ">=")
    s = s.replace("≠", "!=")
    s = s.replace("＝", "=")

    # 绝对值：把 |...| 转成 Abs(...)
    if "|" in s:
        pattern = re.compile(r"\|([^|]+?)\|")
        prev = None
        while prev != s:
            prev = s
            s = pattern.sub(r"Abs(\1)", s)

    # 对数：log_2(x) -> log(x,2)
    s = re.sub(r"\blog_([A-Za-z0-9_]+)\s*\(([^()]*)\)", r"log(\2,\1)", s)

    # 常见函数名
    s = re.sub(r"\bln\b", "log", s)
    return s


def build_local_dict(expr_text: str) -> Dict[str, object]:
    """
    为 parse_expr 构造 local_dict：
    - 自动创建出现在表达式中的符号变量（如 x, y, a1）
    - 提供常用函数/常量
    """
    expr_text = normalize_math_text(expr_text)

    # 变量名：允许字母/下划线/数字（首字符必须是字母/下划线）
    names = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", expr_text))

    # 排除函数名/常量名（避免把 sin 当变量）
    reserved = {
        "sin", "cos", "tan", "cot", "sec", "csc",
        "asin", "acos", "atan",
        "sinh", "cosh", "tanh",
        "log", "ln", "exp", "sqrt",
        "pi", "E",
    }
    vars_ = sorted(n for n in names if n not in reserved)

    local_dict: Dict[str, object] = {}
    for v in vars_:
        local_dict[v] = sp.Symbol(v, real=True)

    # 常用函数
    local_dict.update({
        "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
        "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
        "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
        "log": sp.log, "exp": sp.exp, "sqrt": sp.sqrt,
        "pi": sp.pi, "E": sp.E,
        "Abs": sp.Abs,
    })
    return local_dict


def parse_expr_safe(expr_text: str) -> sp.Expr:
    """安全解析表达式：支持 2x、(x+1)(x-1)、a^2 等常见写法。"""
    expr_text = normalize_math_text(expr_text)
    local_dict = build_local_dict(expr_text)
    return parse_expr(expr_text, local_dict=local_dict, transformations=_TRANSFORMS, evaluate=True)




def parse_expr_with_local_dict(expr_text: str, local_dict: Dict[str, object]) -> sp.Expr:
    """使用外部提供的 local_dict 解析表达式（用于方程组共享同一套符号对象）。"""
    expr_text = normalize_math_text(expr_text)
    return parse_expr(expr_text, local_dict=local_dict, transformations=_TRANSFORMS, evaluate=True)


def parse_equation_safe(left: str, right: str) -> sp.Eq:
    """安全解析等式 left = right"""
    return sp.Eq(parse_expr_safe(left), parse_expr_safe(right))


def verify_solution_substitution(
    equations: Iterable[sp.Eq],
    sol: Dict[sp.Symbol, sp.Expr],
) -> Tuple[bool, List[str]]:
    """
    将解代回方程组检查 residual 是否为 0。
    返回：(是否全部通过, 逐条检查信息)
    """
    msgs: List[str] = []
    ok_all = True
    for i, eq in enumerate(equations):
        residual = sp.simplify((eq.lhs - eq.rhs).subs(sol))
        ok = (residual == 0)
        ok_all = ok_all and ok
        msgs.append(f"check#{i}: residual = {residual} -> {'OK' if ok else 'FAIL'}")
    return ok_all, msgs


def verify_identity_by_simplify(lhs: sp.Expr, rhs: sp.Expr) -> bool:
    """恒等式验证：simplify(lhs - rhs) == 0"""
    return sp.simplify(lhs - rhs) == 0


def verify_identity_by_random_points(
    lhs: sp.Expr,
    rhs: sp.Expr,
    symbols: List[sp.Symbol],
    trials: int = 5,
    low: int = -5,
    high: int = 5,
) -> Tuple[bool, List[str]]:
    """
    随机代入验证（辅助）：多点抽样检查 lhs==rhs
    注意：随机代入不能替代严格证明，但能抓住大多数错误。
    """
    msgs: List[str] = []
    if not symbols:
        # 没变量就直接比值
        ok = sp.simplify(lhs - rhs) == 0
        return ok, [f"no-symbols simplify -> {'OK' if ok else 'FAIL'}"]

    for t in range(trials):
        sub = {s: random.randint(low, high) for s in symbols}
        try:
            lv = sp.N(lhs.subs(sub))
            rv = sp.N(rhs.subs(sub))
            ok = bool(sp.Abs(lv - rv) < 1e-9)
        except Exception as e:
            msgs.append(f"trial#{t}: error {e}")
            ok = False
        msgs.append(f"trial#{t}: sub={sub}  lhs={lv} rhs={rv} -> {'OK' if ok else 'FAIL'}")
        if not ok:
            return False, msgs
    return True, msgs


# =========================
# 额外：从模型输出里抽取“答案：...”并做等价判断（用于评测/闭环）
# =========================
_FINAL_ANSWER_RE = re.compile(r"(答案\s*[:：]\s*)(.+)", re.IGNORECASE)

def extract_final_answer_line(text: str) -> Optional[str]:
    """抽取最后一次出现的“答案：xxx”里的 xxx；找不到返回 None。"""
    if not text:
        return None
    m = None
    for m in _FINAL_ANSWER_RE.finditer(text):
        pass
    if not m:
        return None
    return (m.group(2) or "").strip()

def expressions_equivalent(a: str, b: str) -> bool:
    """
    尽量判断两个表达式/数值是否等价：
    - 支持隐式乘法（2x）
    - 失败则退回到“去空格后字符串相等”
    """
    try:
        la = build_local_dict(a)
        lb = build_local_dict(b)
        ea = parse_expr_with_local_dict(a, la)
        eb = parse_expr_with_local_dict(b, lb)
        return bool(sp.simplify(ea - eb) == 0)
    except Exception:
        return a.replace(" ", "") == b.replace(" ", "")
