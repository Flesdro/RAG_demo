from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS


#把docs变成字符串列表
def load_docs_texts(docs_dir: str = "docs"):
    texts = [] #创建空字符串列表
    for p in Path(docs_dir).rglob("*"): #遍历docs文件夹下的所有文件
        if p.suffix.lower() in {".txt", ".md"}: #如果文件后缀是.txt或.md
            texts.append(p.read_text(encoding="utf-8", errors="ignore")) #读取文件内容并添加到texts列表中
    if not texts: #如果texts列表为空
        raise RuntimeError("docs/ 里没有 .txt 或 .md 文件，先放点文本进去。") #抛出异常
    return texts 


# 构建RAG链
def build_rag_chain():
    # 1) 读取文档
    texts = load_docs_texts("docs")

    # 2) 切块
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120) #创建切块器，每块大概800字符左右，相邻块有120字符重叠
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t)) #对每个文本进行切块并添加到chunks列表中

    # 3) Embedding + 向量库（FAISS）
    # 能不能检索准，很大程度决定你后面会不会“幻觉”
    emb = OllamaEmbeddings(model="nomic-embed-text") #创建嵌入器，使用nomic-embed-text模型，文本->向量
    vs = FAISS.from_texts(chunks, embedding=emb) #创建FAISS向量库，使用嵌入器将切块文本转换为向量并存储
    retriever = vs.as_retriever(search_kwargs={"k": 100}) #创建向量库的检索器，每次搜索返回4个最相关的切块

    # 4) 本地 LLM（Ollama）
    llm = ChatOllama(model="qwen2.5", temperature=0) # temperature=0 让输出更稳定、更少瞎发挥

    # 5) 约束提示词：只能依据检索内容回答
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是严谨助手。只能依据<context>中的内容回答；如果没有依据就说“资料中没有找到”。"),
        ("human", "问题：{input}\n\n<context>\n{context}\n</context>")
    ])
    # ("system", "...")：系统消息，最高优先级，告诉模型“规则是什么”
    # ("human", "...")：用户消息，包含问题和检索到的上下文内容
    # LangChain 会在你调用链的时候，把 {input} 和 {context} 这些占位符替换成真实内容，形成最终发给模型的 messages。

    doc_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, doc_chain)
    return rag_chain


def main():
    rag_chain = build_rag_chain()

    print("RAG ready. 输入问题，quit 退出。\n")
    while True:
        q = input("Q> ").strip()
        if not q:
            continue
        if q.lower() in {"quit", "exit"}:
            break

        out = rag_chain.invoke({"input": q})
        print("\nA>", out.get("answer", ""))

        # 可选：调试看看检索到了哪些片段
        ctx = out.get("context", [])
        print(f"\n[debug] Retrieved {len(ctx)} chunks")
        for i, d in enumerate(ctx):
            print(f"--- chunk {i} ---")
            print((d.page_content or "")[:300])
        print()


if __name__ == "__main__":
    main()
