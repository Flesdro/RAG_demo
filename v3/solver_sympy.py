from __future__ import annotations
"""
solver_sympy.py
=========================
新增：基于 Sympy 的“推理引擎”模块
功能：
- 尝试解析并求解常见数学题（小学/初中/高中）
- 输出 tool_result（JSON 可序列化），作为 LLM 解释时的“真值依据”
覆盖的题型（尽量通用）：
- arithmetic：纯算式求值
- simplify：化简
- factor：因式分解
- equation：解单个方程（含一元二次等）
- system：解方程组（多元多式）
- derivative：求导
- extrema：导数+极值/单调（给出关键点与基本结论；复杂函数可能退化成只给关键点）
注意：
- 解析失败/题型不支持时返回 None，让主程序回退到纯 RAG。
"""
import re
from typing import Dict, List, Optional, Tuple, Any
import sympy as sp
from verifier import (
    normalize_math_text,
    build_local_dict,
    parse_expr_with_local_dict,
    verify_solution_substitution,
)
def make_template_query(style_level: str, tool_type: str) -> str:
    """用于在知识库里检索“讲解模板/常错点”的查询词（不依赖具体题目）。"""
    # 你可以按自己的 docs 内容进一步改这里的词
    base = {
        "arithmetic": "四则运算 步骤 常错点",
        "simplify": "化简 步骤 常错点",
        "factor": "因式分解 步骤 常错点",
        "equation": "解方程 步骤 常错点",
        "system": "解方程组 步骤 常错点",
        "derivative": "求导 步骤 常错点",
        "extrema": "导数 极值 单调区间 步骤 常错点",
        "unknown": "数学 解题 步骤 常错点",
    }.get(tool_type, "数学 解题 步骤 常错点")
    return f"{style_level} {base}"
def _extract_after_keyword(q: str, keywords: List[str]) -> Optional[str]:
    for kw in keywords:
        idx = q.find(kw)
        if idx != -1:
            return q[idx + len(kw):].strip(" :：\t")
    return None
def _maybe_extract_expr(q: str) -> Optional[str]:
    """提取表达式：优先从 y= / f(x)= 里取，其次取冒号后内容。"""
    qn = normalize_math_text(q)
    m = re.search(r"(?:y|f\s*\(\s*x\s*\))\s*=\s*(.+)$", qn)
    if m:
        return m.group(1).strip()
    # 形如：求导：ln(x)/x
    if ":" in qn:
        return qn.split(":", 1)[1].strip()
    return None
def classify_question(q: str) -> str:
    q = q.strip()
    if any(k in q for k in ["求导", "导数"]):
        if any(k in q for k in ["极值", "最值", "单调", "单调区间"]):
            return "extrema"
        return "derivative"
    if "方程组" in q or q.count("=") >= 2:
        return "system"
    if any(k in q for k in ["解方程", "方程"]) and "=" in q:
        return "equation"
    if any(k in q for k in ["因式分解", "分解"]):
        return "factor"
    if "化简" in q:
        return "simplify"
    # arithmetic：尽量保守（只在没有字母变量时才判）
    qn = normalize_math_text(q)
    # 提取可能的算式部分
    expr = _maybe_extract_expr(qn) or qn
    if re.fullmatch(r"[0-9\.\+\-\*\/\(\)\s]+", expr):
        return "arithmetic"
    return "unknown"
def _solve_arithmetic(q: str) -> Dict[str, Any]:
    expr_text = _maybe_extract_expr(q) or q
    expr = parse_expr_safe(expr_text)
    val = sp.N(expr)
    return {
        "type": "arithmetic",
        "engine": "sympy",
        "expr": str(expr),
        "result": str(val),
        "steps": [
            f"将算式写成标准形式：{expr}",
            f"按运算顺序计算，得到：{val}",
        ],
        "checks": [],
    }
def _solve_simplify(q: str) -> Dict[str, Any]:
    expr_text = _maybe_extract_expr(q) or _extract_after_keyword(q, ["化简"]) or q
    expr = parse_expr_safe(expr_text)
    simp = sp.simplify(expr)
    return {
        "type": "simplify",
        "engine": "sympy",
        "expr": str(expr),
        "result": str(simp),
        "steps": [
            f"原式：{expr}",
            f"化简得到：{simp}",
        ],
        "checks": [],
    }
def _solve_factor(q: str) -> Dict[str, Any]:
    expr_text = _maybe_extract_expr(q) or _extract_after_keyword(q, ["因式分解", "分解"]) or q
    expr = parse_expr_safe(expr_text)
    fac = sp.factor(expr)
    return {
        "type": "factor",
        "engine": "sympy",
        "expr": str(expr),
        "result": str(fac),
        "steps": [
            f"原式：{expr}",
            f"因式分解：{fac}",
        ],
        "checks": [],
    }
def _split_equation_segments(q: str) -> List[str]:
    qn = normalize_math_text(q)
    # 去掉外层花括号等
    qn = qn.replace("{", " ").replace("}", " ")
    # 按常见分隔符切
    segs = re.split(r"[;\n,，]+", qn)
    segs = [s.strip() for s in segs if "=" in s]
    cleaned: List[str] = []
    for s in segs:
        # 新增：去掉等式左侧可能出现的中文提示词（例如：解方程:  求解:）
        if ":" in s and s.find(":") < s.find("="):
            s = s.split(":", 1)[1].strip()
        cleaned.append(s)
    return cleaned
def _parse_equations(segs: List[str]) -> List[sp.Eq]:
    eqs: List[sp.Eq] = []
    for s in segs:
        if "=" not in s:
            continue
        left, right = s.split("=", 1)
        left = left.strip()
        right = right.strip()
        if not left or not right:
            continue
        eqs.append(parse_equation_safe(left, right))
    return eqs


def _parse_equations_shared(segs: List[str]) -> Tuple[List[sp.Eq], Dict[str, object]]:
    """新增：为方程组构造共享 local_dict，确保 x/y 等符号对象一致。"""
    joined = " ; ".join(segs)
    local_dict = build_local_dict(joined)
    eqs: List[sp.Eq] = []
    for s in segs:
        if "=" not in s:
            continue
        left, right = s.split("=", 1)
        left = left.strip()
        right = right.strip()
        if not left or not right:
            continue
        lhs = parse_expr_with_local_dict(left, local_dict)
        rhs = parse_expr_with_local_dict(right, local_dict)
        eqs.append(sp.Eq(lhs, rhs))
    return eqs, local_dict

def _solve_equation(q: str) -> Dict[str, Any]:
    segs = _split_equation_segments(q)
    if not segs:
        # 兜底：尝试从整句里找第一个等式
        m = re.search(r"(.+?)=(.+)", normalize_math_text(q))
        if not m:
            raise ValueError("no equation found")
        segs = [m.group(0)]
    eq = _parse_equations(segs)[0]
    syms = sorted(eq.free_symbols, key=lambda s: s.name)
    # 默认解第一个变量（常见是一元方程 x）
    vars_ = syms if syms else [sp.Symbol("x")]
    sols = sp.solve(eq, vars_[0] if len(vars_) == 1 else vars_, dict=True)
    checks: List[str] = []
    if sols:
        ok, msgs = verify_solution_substitution([eq], sols[0])
        checks.extend(msgs)
        checks.insert(0, f"substitution_check={'OK' if ok else 'FAIL'}")
    return {
        "type": "equation",
        "engine": "sympy",
        "equation": f"{eq.lhs} = {eq.rhs}",
        "variables": [v.name for v in vars_],
        "result": str(sols),
        "steps": [
            f"将题目写成方程：{eq.lhs} = {eq.rhs}",
            "用代数方法求解（sympy.solve）得到解集。",
        ],
        "checks": checks,
    }
def _solve_system(q: str) -> Dict[str, Any]:
    segs = _split_equation_segments(q)
    eqs, _ld = _parse_equations_shared(segs)
    if len(eqs) < 2:
        # 如果只有一个等式，降级为 equation
        return _solve_equation(q)
    # 收集变量
    symbols = sorted(set().union(*[eq.free_symbols for eq in eqs]), key=lambda s: s.name)
    sols = sp.solve(eqs, symbols, dict=True)
    checks: List[str] = []
    if sols:
        ok, msgs = verify_solution_substitution(eqs, sols[0])
        checks.extend(msgs)
        checks.insert(0, f"substitution_check={'OK' if ok else 'FAIL'}")
    return {
        "type": "system",
        "engine": "sympy",
        "equations": [f"{eq.lhs} = {eq.rhs}" for eq in eqs],
        "variables": [s.name for s in symbols],
        "result": str(sols),
        "steps": [
            "将题目整理为方程组：",
            *[f"- {eq.lhs} = {eq.rhs}" for eq in eqs],
            "用代数方法联立求解（sympy.solve）得到解集。",
        ],
        "checks": checks,
    }
def _solve_derivative(q: str) -> Dict[str, Any]:
    expr_text = _maybe_extract_expr(q) or _extract_after_keyword(q, ["求导", "导数"]) or q
    expr = parse_expr_safe(expr_text)
    x = sp.Symbol("x", real=True)
    der = sp.diff(expr, x)
    return {
        "type": "derivative",
        "engine": "sympy",
        "expr": str(expr),
        "result": str(der),
        "steps": [
            f"设 y = {expr}",
            f"对 x 求导，得到 y' = {der}",
        ],
        "checks": [],
    }
def _solve_extrema(q: str) -> Dict[str, Any]:
    expr_text = _maybe_extract_expr(q) or _extract_after_keyword(q, ["求导", "导数"]) or q
    expr = parse_expr_safe(expr_text)
    x = sp.Symbol("x", real=True)
    der = sp.diff(expr, x)
    critical = sp.solve(sp.Eq(der, 0), x)
    steps = [
        f"设 y = {expr}",
        f"求导：y' = {der}",
        f"令 y' = 0，解得临界点：{critical}",
    ]
    # 尝试用二阶导判断（不保证所有情况都能简洁判断）
    try:
        der2 = sp.diff(der, x)
        steps.append(f"二阶导：y'' = {der2}")
        extrema_info = []
        for c in critical:
            v = sp.simplify(der2.subs(x, c))
            if v.is_real:
                if v > 0:
                    extrema_info.append(f"x={c} 为极小值点（y''>0）")
                elif v < 0:
                    extrema_info.append(f"x={c} 为极大值点（y''<0）")
                else:
                    extrema_info.append(f"x={c} 二阶导为 0，需进一步判别")
            else:
                extrema_info.append(f"x={c} 需进一步判别")
        if extrema_info:
            steps.append("判别结果：")
            steps.extend([f"- {t}" for t in extrema_info])
    except Exception:
        pass
    return {
        "type": "extrema",
        "engine": "sympy",
        "expr": str(expr),
        "derivative": str(der),
        "critical_points": [str(c) for c in critical],
        "result": f"critical_points={critical}",
        "steps": steps,
        "checks": [],
    }
def solve_math_question(q: str) -> Optional[Dict[str, Any]]:
    """
    尝试用 Sympy 解题。
    返回 tool_result dict（可 json 序列化）；失败返回 None。
    """
    qn = normalize_math_text(q)
    typ = classify_question(qn)
    try:
        if typ == "arithmetic":
            return _solve_arithmetic(qn)
        if typ == "simplify":
            return _solve_simplify(qn)
        if typ == "factor":
            return _solve_factor(qn)
        if typ == "equation":
            return _solve_equation(qn)
        if typ == "system":
            return _solve_system(qn)
        if typ == "derivative":
            return _solve_derivative(qn)
        if typ == "extrema":
            return _solve_extrema(qn)
        return None
    except Exception:
        # 解析/求解失败就回退到纯 RAG
        return None