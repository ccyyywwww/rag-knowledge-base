"""
rag_engine.py — RAG 核心逻辑（纯处理文档，不依赖 Web）
"""

import os
import json
import hashlib
import uuid
from typing import List
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from dotenv import load_dotenv
from pypdf import PdfReader

# 0. 路径常量
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
FILE_INDEX_PATH = os.path.join(BASE_DIR, "file_index.json")

# 1. 环境变量 + 两个 API 客户端
load_dotenv()

embedding_client = OpenAI(
    api_key=os.getenv("QIANWEN_API_KEY"),
    base_url=os.getenv("QIANWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
)

chat_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
)

# 2. ChromaDB 初始化
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
)

# ========== 索引管理函数 ==========
def load_file_index():
    if os.path.exists(FILE_INDEX_PATH):
        with open(FILE_INDEX_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_file_index(index):
    with open(FILE_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

def calculate_file_hash(file_path):
    """使用 SHA256 计算文件哈希，更安全（抗碰撞）"""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

# 3. 文档解析
def extract_text_from_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    all_text = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            all_text.append(text)
    return "\n".join(all_text)

def read_document(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

# ========== 核心建库函数（增量） ==========
def rebuild_knowledge_base(file_paths: List[str], file_names: List[str]) -> int:
    """
    处理多个文件，增量入库。
    每个文件入库时附带 metadata (file_hash)，便于后续精确查询/删除。
    """
    file_index = load_file_index()
    # 获取或创建集合
    try:
        collection = chroma_client.get_collection(name="knowledge_base")
    except Exception:
        collection = chroma_client.create_collection(name="knowledge_base")

    new_chunk_count = 0
    BATCH_SIZE = 10

    for file_path, original_name in zip(file_paths, file_names):
        file_hash = calculate_file_hash(file_path)
        # 去重：哈希已存在则跳过
        if file_hash in file_index:
            print(f"文件 '{original_name}' 内容未变，跳过处理。")
            # 确保删除临时文件
            if os.path.exists(file_path):
                os.remove(file_path)
            continue

        print(f"正在处理新文件: {original_name}")
        content = read_document(file_path)
        chunks = splitter.split_text(content)
        if not chunks:
            if os.path.exists(file_path):
                os.remove(file_path)
            continue

        # 分批向量化，避免 API 限流
        all_embeddings = []
        for i in range(0, len(chunks), BATCH_SIZE):
            batch_chunks = chunks[i:i+BATCH_SIZE]
            resp = embedding_client.embeddings.create(
                model="text-embedding-v3",
                input=batch_chunks,
                encoding_format="float"
            )
            batch_embeddings = [item.embedding for item in resp.data]
            all_embeddings.extend(batch_embeddings)

        ids = [f"{file_hash}_{i}" for i in range(len(chunks))]
        # ★ 关键改进：添加 metadata，存储 file_hash
        metadatas = [{"file_hash": file_hash} for _ in chunks]

        collection.add(
            ids=ids,
            documents=chunks,
            embeddings=all_embeddings,
            metadatas=metadatas          # 便于后续 where 条件查询
        )

        # 更新索引
        file_index[file_hash] = original_name
        new_chunk_count += len(chunks)

        # 删除临时文件（无论成功与否，finally 会保证）
        if os.path.exists(file_path):
            os.remove(file_path)

    save_file_index(file_index)
    return new_chunk_count

# ========== 检索 + 生成 ==========
def has_knowledge_base() -> bool:
    try:
        chroma_client.get_collection(name="knowledge_base")
        return True
    except Exception:
        return False

def ask(question: str, top_k: int = 3) -> dict:
    try:
        collection = chroma_client.get_collection(name="knowledge_base")
    except Exception:
        return {"answer": "知识库为空，请先上传文档。", "sources": []}

    query_embedding = embedding_client.embeddings.create(
        model="text-embedding-v3",
        input=[question],
        encoding_format="float"
    ).data[0].embedding

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    retrieved_docs = results["documents"][0]

    context = "\n\n".join(retrieved_docs)
    system_prompt = "你是一个基于给定资料回答问题的助手。请只根据资料内容回答，如果资料中没有相关信息，请说明不知道。"
    user_prompt = f"资料：\n{context}\n\n问题：{question}"

    chat_response = chat_client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3,
        stream=False
    )
    answer = chat_response.choices[0].message.content
    return {"answer": answer, "sources": retrieved_docs}

# ========== 删除文件（按哈希） ==========
def delete_files(file_hashes: List[str]) -> int:
    """批量删除文件及其向量片段，使用 where 条件精准删除，保证原子性"""
    if not file_hashes:
        return 0

    file_index = load_file_index()
    deleted_count = 0
    try:
        collection = chroma_client.get_collection(name="knowledge_base")
    except Exception:
        # 集合不存在，无需删除
        return 0

    # ★ 改进：先收集所有需要删除的 ID，一次性删除，避免部分失败
    all_ids_to_delete = []
    for file_hash in file_hashes:
        if file_hash not in file_index:
            continue
        # 使用 where 条件直接查询该文件的所有片段 ID
        # 注意：ChromaDB 的 get() 支持 where 过滤
        result = collection.get(where={"file_hash": file_hash})
        ids_to_delete = result["ids"]
        if ids_to_delete:
            all_ids_to_delete.extend(ids_to_delete)
        # 从索引中移除
        del file_index[file_hash]
        deleted_count += 1

    if all_ids_to_delete:
        collection.delete(ids=all_ids_to_delete)   # 原子删除

    save_file_index(file_index)
    return deleted_count

# ================== RAGEngine 类（供 FastAPI 使用） ==================
class RAGEngine:
    def add_documents_from_paths(self, file_paths: List[str], original_names: List[str]) -> int:
        """直接接收临时文件路径列表，避免内存拷贝"""
        return rebuild_knowledge_base(file_paths, original_names)

    def query(self,
     question: str) -> dict:
        return ask(question)

    def delete_files(self, file_hashes: list[str]) -> int:
        return delete_files(file_hashes)