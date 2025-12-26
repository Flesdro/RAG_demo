import json
import re
import numpy as np
from pathlib import Path

import rag_v7_step_tree_intent_tone_linked as rag

SAVE_FULL_ANSWER = False
REFUSE_TEXT = "资料中没有找到"
NEED_CALC_PAT = re.compile(r"(求值|求解|解方程|求导|化简|计算|最值|极值|面积|周长|体积|概率|方程|不等式|交点|解集|几何|函数)", re.I)

USE_ROUTER = getattr(rag, "USE_ROUTER", False)
USE_MMR_DEFAULT = getattr(rag, "USE_MMR_DEFAULT", True)

solve_math_question = getattr(rag, "solve_math_question", None)
HAS_SOLVER = callable(solve_math_question)

def to_jsonable(obj):
    if obj is None: return None
    if isinstance(obj, (np.generic,)): return obj.item()
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict): return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [to_jsonable(x) for x in obj]
    return obj

def append_jsonl(path, row):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                yield json.loads(line)

def extract_final_answer(text: str) -> str:
    if not text: return ""
    m = re.search(r"答案：\s*(.*)", text)
    return m.group(1).strip() if m else ""

def tool_repair_question(q: str) -> str:
    s = q.strip()
    # 去掉 leading 前缀：计算：/求值：/解：/求导：
    s = re.sub(r"^\s*(计算|求值|求|解|化简|求导)\s*[:：]\s*", "", s)
    # 去掉问号
    s = s.replace("？", "").replace("?", "")
    # 把中文符号换成算符
    s = s.replace("＝", "=").replace("×", "*").replace("÷", "/")
    # 处理 “... = ” 这类（例如 37 + 58 =）
    s = re.sub(r"\s*=\s*$", "", s)
    # 去掉尾巴（逗号后“求/解/计算...”）
    s = re.sub(r"[，,;；。]\s*(求|解|计算|化简).*$", "", s)
    # 去括号条件（(x>0)）
    s = re.sub(r"\([^)]*\)", "", s)
    # 自然语言比较符
    s = s.replace("大于等于", ">=").replace("小于等于", "<=")
    s = s.replace("不等于", "!=")
    return s.strip()

def _format_solutions_for_answer(tool_result: dict) -> str:
    if not isinstance(tool_result, dict): return ""
    if tool_result.get("final_answer"): return str(tool_result["final_answer"]).strip()
    if "result" in tool_result and tool_result.get("type") not in ("equation","system","inequality"):
        return str(tool_result["result"]).strip()
    sols = tool_result.get("solutions")
    if isinstance(sols, list) and sols:
        items=[]
        for s in sols:
            if isinstance(s, dict) and s:
                items.append(", ".join([f"{k}={v}" for k,v in s.items()]))
        if items: return " 或 ".join(items)
    if "result" in tool_result: return str(tool_result["result"]).strip()
    return ""

def enforce_final_answer(answer_text: str, expected_final: str):
    if not expected_final: return answer_text, False
    if re.search(r"答案：", answer_text):
        new_text = re.sub(r"(答案：\s*)(.*)", rf"\1{expected_final}", answer_text, count=1)
        return new_text, (new_text != answer_text)
    return (answer_text.rstrip() + f"\n答案：{expected_final}\n"), True

def run_one(llm, router_llm, planner_llm, stores, embeddings, item,
            use_solver=True, enable_step_tree=True, step_tree_eval=True,
            auto_level=False, tone="kind"):

    q = item.get("question","").strip()
    chosen_level = item.get("level","middle")

    route_level_fn = getattr(rag, "route_level", None)
    routed = route_level_fn(router_llm, q) if (USE_ROUTER and callable(route_level_fn)) else chosen_level

    retrieval_level = routed if auto_level else chosen_level
    style_level = chosen_level

    tool_result = None
    q_try = q

    # 1) tool try
    if use_solver and HAS_SOLVER:
        try:
            tool_result = solve_math_question(q_try)
        except Exception:
            tool_result = None

    # 2) tool repair retry
    if tool_result is None and use_solver and HAS_SOLVER:
        q2 = tool_repair_question(q)
        if q2 and q2 != q:
            try:
                tool_result = solve_math_question(q2)
                q_try = q2
            except Exception:
                tool_result = None

    # tool mode
    if tool_result is not None:
        tq = rag.make_template_query(style_level, tool_result.get("type","unknown"))
        tdocs = rag.retrieve_with_filter(stores[style_level], embeddings, tq, use_mmr=USE_MMR_DEFAULT)

        step_tree_obj = {}
        if enable_step_tree and callable(getattr(rag, "build_step_tree", None)):
            try:
                tmp = rag.build_step_tree(planner_llm, q_try, tool_result, hint_docs=tdocs)
                step_tree_obj = tmp or {}
                if step_tree_eval and step_tree_obj and callable(getattr(rag, "eval_step_tree_inplace", None)):
                    step_tree_obj = rag.eval_step_tree_inplace(step_tree_obj)
            except Exception as e:
                # ✅ step_tree 失败不应影响“计算答案”
                step_tree_obj = {"error": repr(e)}

        prompt = rag.build_tool_prompt(style_level, tone)
        chain = rag.create_stuff_documents_chain(llm, prompt)
        out = chain.invoke({
            "input": q_try,
            "tool": json.dumps(tool_result, ensure_ascii=False),
            "step_tree": json.dumps(step_tree_obj, ensure_ascii=False),
            "context": tdocs
        })
        answer_text = out if isinstance(out, str) else out.get("output_text", str(out))

        expected_final = _format_solutions_for_answer(tool_result)
        answer_text, fixed = enforce_final_answer(answer_text, expected_final)
        final_extracted = extract_final_answer(answer_text)

        res_min = {
            "id": item.get("id"),
            "level": chosen_level,
            "routed_level": routed,
            "question": q,
            "question_used_for_tool": q_try,
            "final_answer": final_extracted or expected_final,
            "used_mode": "tool",
            "fixed_final": bool(fixed),
            "step_tree_ok": ("error" not in step_tree_obj),
        }
        if SAVE_FULL_ANSWER:
            res_min["answer"] = answer_text
            res_min["tool_result"] = tool_result
            res_min["step_tree"] = step_tree_obj
        return res_min, answer_text

    # rag-only refuse gate
    if NEED_CALC_PAT.search(q):
        answer_text = f"问题：{q}\n\n{REFUSE_TEXT}\n答案：{REFUSE_TEXT}"
        res_min = {
            "id": item.get("id"),
            "level": chosen_level,
            "routed_level": routed,
            "question": q,
            "final_answer": REFUSE_TEXT,
            "used_mode": "rag_only_refuse",
            "rag_only_reason": "need_calc_but_no_tool",
        }
        if SAVE_FULL_ANSWER:
            res_min["answer"] = answer_text
        return res_min, answer_text

    # normal rag-only
    docs = rag.retrieve_with_filter(stores[retrieval_level], embeddings, q, use_mmr=USE_MMR_DEFAULT)
    prompt = rag.build_prompt(style_level, tone)
    chain = rag.create_stuff_documents_chain(llm, prompt)
    out = chain.invoke({"input": q, "context": docs})
    answer_text = out if isinstance(out, str) else out.get("output_text", str(out))

    res_min = {
        "id": item.get("id"),
        "level": chosen_level,
        "routed_level": routed,
        "question": q,
        "final_answer": extract_final_answer(answer_text),
        "used_mode": "rag_only",
        "rag_only_reason": "no_tool_result",
    }
    if SAVE_FULL_ANSWER:
        res_min["answer"] = answer_text
        res_min["docs"] = [d.metadata for d in docs]
    return res_min, answer_text

if __name__ == "__main__":
    in_path = "test/primary_15.jsonl"
    out_path = "test_result/pred_primary_15_min.jsonl"

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    open(out_path, "w", encoding="utf-8").close()

    embeddings = rag.OllamaEmbeddings(model=rag.EMBED_MODEL)
    llm = rag.ChatOllama(model=rag.LLM_MODEL, temperature=rag.TEMPERATURE)
    router_llm = rag.ChatOllama(model=rag.LLM_MODEL, temperature=0)
    planner_llm = rag.ChatOllama(model=rag.LLM_MODEL, temperature=0)
    stores = {k: rag.load_or_build_vectorstore(cfg, embeddings) for k, cfg in rag.LEVELS.items()}

    for idx, item in enumerate(read_jsonl(in_path), start=1):
        try:
            res_min, answer_text = run_one(
                llm=llm, router_llm=router_llm, planner_llm=planner_llm,
                stores=stores, embeddings=embeddings, item=item,
                use_solver=True, enable_step_tree=True, step_tree_eval=True,
                auto_level=False, tone="kind",
            )
            print("=" * 80)
            print(f"[{idx}] id={res_min.get('id')} level={res_min.get('level')} mode={res_min.get('used_mode')} step_tree_ok={res_min.get('step_tree_ok', True)} tool_q={res_min.get('question_used_for_tool','')}")
            print("Q:", res_min.get("question"))
            print("A(full):\n", answer_text)
            print("A(saved_final):", res_min.get("final_answer") or "(未抽取到“答案：”行)")
            append_jsonl(out_path, res_min)
        except Exception as e:
            err_row = {"id": item.get("id"), "level": item.get("level"), "question": item.get("question"), "error": repr(e)}
            append_jsonl(out_path, err_row)
            print(f"[{idx}] ERROR on id={item.get('id')}: {e}")

    print("\nDone ->", out_path)