import json
import re
import numpy as np
from pathlib import Path

# 直接 import 你的主程序模块（不要运行 main）
import rag_v6_step_tree_intent_tone as rag

def to_jsonable(obj):
    """把 numpy 标量/数组等转成 JSON 可写类型"""
    if obj is None:
        return None
    if isinstance(obj, (np.generic,)):      # np.float32 / np.int64 ...
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, set):
        return [to_jsonable(x) for x in obj]
    return obj

def append_jsonl(path, row):
    """每处理一题就追加写一行 jsonl"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def extract_final_answer(text: str) -> str:
    """从模型输出里抓 ‘答案：xxx’（你已规定了输出格式）"""
    if not text:
        return ""
    m = re.search(r"答案：\s*(.*)", text)
    return m.group(1).strip() if m else ""

def run_one(
    llm,
    router_llm,
    planner_llm,
    stores,
    embeddings,
    item,
    use_solver=True,
    enable_step_tree=True,
    step_tree_eval=True,
    auto_level=False,
    tone="kind",
):
    q = item["question"]
    chosen_level = item.get("level", "middle")  # 默认用数据集标注的 level
    routed = rag.llm_route_level(router_llm, q) if auto_level else chosen_level

    # 解耦：检索档 & 表达档
    retrieval_level = routed if auto_level else chosen_level
    style_level = chosen_level

    tool_result = None
    if use_solver and rag.HAS_SOLVER:
        tool_result = rag.solve_math_question(q)

    # ========== Tool 模式 ==========
    if tool_result is not None:
        tq = rag.make_template_query(style_level, tool_result.get("type", "unknown"))
        tdocs = rag.retrieve_with_filter(stores[style_level], embeddings, tq, use_mmr=rag.USE_MMR_DEFAULT)

        step_tree_obj = {}
        if enable_step_tree:
            tmp = rag.build_step_tree(planner_llm, q, tool_result, hint_docs=tdocs)
            step_tree_obj = tmp or {}
            if step_tree_eval and step_tree_obj:
                step_tree_obj = rag.eval_step_tree_inplace(step_tree_obj)

        prompt = rag.build_tool_prompt(style_level, tone)
        chain = rag.create_stuff_documents_chain(llm, prompt)
        out = chain.invoke({
            "input": q,
            "tool": json.dumps(tool_result, ensure_ascii=False),
            "step_tree": json.dumps(step_tree_obj, ensure_ascii=False),
            "context": tdocs
        })
        answer_text = out if isinstance(out, str) else out.get("output_text", str(out))

        return {
            "id": item.get("id"),
            "level": chosen_level,
            "routed_level": routed,
            "question": q,
            "answer": answer_text,
            "final_extracted": extract_final_answer(answer_text),
            "tool_result": tool_result,
            "step_tree": step_tree_obj,
            "used_mode": "tool"
        }

    # ========== RAG-only 模式 ==========
    docs = rag.retrieve_with_filter(stores[retrieval_level], embeddings, q, use_mmr=rag.USE_MMR_DEFAULT)
    prompt = rag.build_prompt(style_level, tone)
    chain = rag.create_stuff_documents_chain(llm, prompt)
    out = chain.invoke({"input": q, "context": docs})
    answer_text = out if isinstance(out, str) else out.get("output_text", str(out))

    return {
        "id": item.get("id"),
        "level": chosen_level,
        "routed_level": routed,
        "question": q,
        "answer": answer_text,
        "final_extracted": extract_final_answer(answer_text),
        "docs": [d.metadata for d in docs],
        "used_mode": "rag_only"
    }


if __name__ == "__main__":
    in_path = "test/high_15.jsonl"
    out_path = "test_result/pred_high_15.jsonl"

    # 先清空旧输出（避免重复追加）
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    open(out_path, "w", encoding="utf-8").close()

    # 1) 初始化（与你原来一样）
    embeddings = rag.OllamaEmbeddings(model=rag.EMBED_MODEL)
    llm = rag.ChatOllama(model=rag.LLM_MODEL, temperature=rag.TEMPERATURE)
    router_llm = rag.ChatOllama(model=rag.LLM_MODEL, temperature=0)
    planner_llm = rag.ChatOllama(model=rag.LLM_MODEL, temperature=0)
    stores = {k: rag.load_or_build_vectorstore(cfg, embeddings) for k, cfg in rag.LEVELS.items()}

    # 2) 逐题跑：跑完立刻打印 + 立刻写文件
    for idx, item in enumerate(read_jsonl(in_path), start=1):
        try:
            result = run_one(
                llm=llm,
                router_llm=router_llm,
                planner_llm=planner_llm,
                stores=stores,
                embeddings=embeddings,
                item=item,
                use_solver=True,
                enable_step_tree=True,
                step_tree_eval=True,
                auto_level=False,
                tone="kind",
            )

            # ✅ 每题打印一个答案
            # ✅ 每题打印完整回答
            print("=" * 80)
            print(f"[{idx}] id={result.get('id')} level={result.get('level')} routed={result.get('routed_level')}")
            print("Q:", result.get("question"))
            print("A(full):\n", result.get("answer", ""))

            # ✅ 只存“最终答案”（以及你想保留的少量字段）
            final_ans = result.get("final_extracted") or ""
            # 可选兜底：抽取不到“答案：”就用 tool_result.result
            if not final_ans and isinstance(result.get("tool_result"), dict):
                final_ans = str(result["tool_result"].get("result", ""))

            save_row = {
                "id": result.get("id"),
                "level": result.get("level"),
                "routed_level": result.get("routed_level"),
                "question": result.get("question"),
                "final_answer": final_ans,
                "used_mode": result.get("used_mode"),
            }

            append_jsonl(out_path, save_row)


        except Exception as e:
            # 出错也记录一下，方便定位是哪一题
            err_row = {
                "id": item.get("id"),
                "level": item.get("level"),
                "question": item.get("question"),
                "error": repr(e),
            }
            append_jsonl(out_path, err_row)
            print(f"[{idx}] ERROR on id={item.get('id')}: {e}")

    print("\nDone ->", out_path)
