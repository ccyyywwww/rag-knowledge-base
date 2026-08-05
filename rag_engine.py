"""
rag_engine.py — RAG 核心逻辑（纯处理文档，不依赖 Web）

这一层只负责四件事：
  1. 解析文档（PDF / TXT）
  2. 切分文本 → 向量化 → 存入 ChromaDB
  3. 用户提问 → 检索最相似的段落
  4. 段落 + 问题 → 让 AI 生成答案

app.py 只管调用这里的函数，完全不懂"切分"和"检索"的细节。
以后你想做命令行版或 Agent 版，直接 import 这个文件复用即可。
"""

import os
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from dotenv import load_dotenv
from pypdf import PdfReader

# ========== 0. 路径常量 ==========
# __file__ 是 Python 内置的魔法变量，永远指向"当前这个文件"。
# abspath 把它变成绝对路径，dirname 取它的文件夹。
# 这样不管你在哪个目录启动程序，都能正确定位文件
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

# ========== 1. 环境变量 + 两个 API 客户端 ==========
load_dotenv()

# Embedding 客户端（阿里云千问）—— DeepSeek 没有 embedding 接口，所以用千问
embedding_client = OpenAI(
    api_key=os.getenv("QIANWEN_API_KEY"),
    base_url=os.getenv("QIANWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
)

# Chat 客户端（DeepSeek）—— 负责最终生成回答
chat_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
)

# ========== 2. ChromaDB 初始化 ==========
# PersistentClient：数据写进硬盘，程序重启后依然存在
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

# 文本切分器
# 分割优先级：段落 > 句子 > 中文标点 > 空格 > 字符
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
)


# ========== 3. 文档解析 ==========
def extract_text_from_pdf(file_path: str) -> str:
    """用 pypdf 逐页读取 PDF 的文字，拼成一个长字符串。"""
    reader = PdfReader(file_path)
    all_text = []
    for page in reader.pages:
        text = page.extract_text()
        if text:  # 扫描版 PDF 没有文字层，extract_text() 会返回空字符串
            all_text.append(text)
    return "\n".join(all_text)


def read_document(file_path: str) -> str:
    """根据扩展名自动选择解析方式，统一返回纯文本。"""
    ext = os.path.splitext(file_path)[1].lower()  # 取文件扩展名，如 ".pdf"
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    else:
        # 非 PDF 一律按 TXT 处理（支持 UTF-8 编码）
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()


# ========== 4. 建库（上传文档时调用） ==========
def rebuild_knowledge_base(file_path: str) -> int:
    """清空旧库 → 解析 → 切分 → 向量化 → 入库。返回切分的片段数。"""
    # 学习阶段简单粗暴：每次上传都重建整个库。
    # 生产环境应该做增量更新（只处理新增/修改的文件）。
    try:
        chroma_client.delete_collection("knowledge_base")
    except Exception:
        pass  # 集合不存在时 delete 会报错，忽略即可
    collection = chroma_client.create_collection(name="knowledge_base")

    # 解析 + 切分
    content = read_document(file_path)
    chunks = splitter.split_text(content)
    if not chunks:
        return 0

    # 批量向量化（一次传所有 chunk，比逐个调用快很多）
    resp = embedding_client.embeddings.create(
        model="text-embedding-v3",
        input=chunks,
        encoding_format="float"
    )
    embeddings = [item.embedding for item in resp.data]

    # 写入 ChromaDB
    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings
    )
    return len(chunks)


# ========== 5. 检索 + 生成（用户提问时调用） ==========
def has_knowledge_base() -> bool:
    """判断知识库是否已建立（是否上传过文档）。"""
    try:
        # get_collection 不存在时抛 ValueError
        chroma_client.get_collection(name="knowledge_base")
        return True
    except Exception:
        return False


def ask(question: str, top_k: int = 3) -> dict:
    """输入用户问题，返回 {'answer': AI 回答, 'sources': 参考段落列表}。"""
    collection = chroma_client.get_collection(name="knowledge_base")

    # 5.1 把问题也转成向量
    query_embedding = embedding_client.embeddings.create(
        model="text-embedding-v3",
        input=[question],
        encoding_format="float"
    ).data[0].embedding

    # 5.2 检索与问题最相似的 top_k 个段落
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    retrieved_docs = results["documents"][0]

    # 5.3 把段落拼成上下文，塞进 prompt
    # 这里最多 3 段 × 500 字 ≈ 1500 字，远低于 token 上限，不需要手动截断
    context = "\n\n".join(retrieved_docs)
    system_prompt = "你是一个基于给定资料回答问题的助手。请只根据资料内容回答，如果资料中没有相关信息，请说明不知道。"
    user_prompt = f"资料：\n{context}\n\n问题：{question}"

    # 5.4 调用 DeepSeek 生成答案（非流式，简单起见）
    chat_response = chat_client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3,  # 较低温度让回答更稳定
        stream=False
    )
    answer = chat_response.choices[0].message.content
    return {"answer": answer, "sources": retrieved_docs}
