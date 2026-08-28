"""
app.py — FastAPI 接口层（Web 入口）

只做三件事：
  1. 把 static/index.html 页面发给浏览器
  2. 接收上传的文档，交给 rag_engine 建库
  3. 接收用户提问，交给 rag_engine 回答

运行方式：
    uvicorn app:app --reload
然后浏览器打开 http://127.0.0.1:8000
"""

import os
import json
from uuid import uuid4
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
from rag_engine import RAGEngine

app = FastAPI(title="我的知识库", version="2.0")
rag_engine = RAGEngine()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

# 确保上传目录存在
os.makedirs(UPLOAD_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

class QuestionRequest(BaseModel):
    question: str

class DeleteRequest(BaseModel):
    hashes: List[str]

# --- API 路由 ---

@app.get("/")
async def read_root():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>static/index.html 未找到</h1>")

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """
    批量上传文件，流式写入临时文件，避免内存溢出。

    安全防护（v2.1 补回）：
    1. 扩展名白名单：仅允许 .pdf / .txt
    2. 每文件 50MB 大小限制：边读边计数，超限即中止
    """
    ALLOWED_EXT = {".pdf", ".txt"}
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

    # ① 先校验扩展名（不涉及文件写入，放在 try 外，错误直接以 400 返回）
    for file in files:
        ext = os.path.splitext(os.path.basename(file.filename))[1].lower()
        if ext not in ALLOWED_EXT:
            raise HTTPException(status_code=400, detail=f"不支持的文件类型：{ext}（仅支持 PDF / TXT）")

    temp_paths = []
    original_names = []
    try:
        for file in files:
            # 生成安全的临时文件名（UUID + 原始基名）
            safe_name = os.path.basename(file.filename)
            temp_filename = f"{uuid4().hex}_{safe_name}"
            temp_path = os.path.join(UPLOAD_DIR, temp_filename)

            # ② 流式写入 + 大小限制：分块读取，超 50MB 立即中止
            size = 0
            with open(temp_path, "wb") as f:
                while True:
                    chunk = await file.read(1024 * 1024)  # 每次 1MB
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > MAX_FILE_SIZE:
                        raise HTTPException(status_code=413, detail=f"文件 {safe_name} 超过 50MB 限制")
                    f.write(chunk)

            temp_paths.append(temp_path)
            original_names.append(safe_name)

        # ③ 调用 RAG 引擎处理（传入路径列表）
        new_chunks = rag_engine.add_documents_from_paths(temp_paths, original_names)
        return {"message": f"成功处理 {len(files)} 个文件，新增 {new_chunks} 个知识片段。"}

    except HTTPException:
        # 超限(413)等：清理已写入的临时文件后重新抛出，保留正确的状态码
        for p in temp_paths:
            if os.path.exists(p):
                os.remove(p)
        raise
    except Exception as e:
        # 其他异常：同样清理后返回 500
        for p in temp_paths:
            if os.path.exists(p):
                os.remove(p)
        print(f"Error during upload: {e}")
        raise HTTPException(status_code=500, detail="文件处理失败")

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    question = request.question
    if not question:
        raise HTTPException(400, "问题不能为空")
    try:
        result = rag_engine.query(question)
        return JSONResponse(content={"answer": result["answer"], "sources": result["sources"]})
    except Exception as e:
        print(f"Error during query: {e}")
        raise HTTPException(500, "回答生成失败")

@app.get("/files")
async def list_files():
    """列出所有已入库文件（哈希 + 原始名）"""
    index_path = os.path.join(BASE_DIR, "file_index.json")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        files = [{"hash": h, "name": name} for h, name in data.items()]
        return {"files": files}
    return {"files": []}

@app.get("/file/{file_hash}")
async def get_file_content(file_hash: str):
    """
    根据文件哈希获取该文件的完整原文（合并所有片段）。
    使用 where 条件直接查询，高效且精准。
    """
    from rag_engine import chroma_client
    try:
        collection = chroma_client.get_collection("knowledge_base")
        # ★ 使用 where 过滤，无需扫描所有 ID
        result = collection.get(where={"file_hash": file_hash})
        docs = result["documents"]
        if not docs:
            raise HTTPException(404, "文件未找到")
        # 按顺序排序（因为 ChromaDB 返回顺序可能与入库顺序不一致，我们按 ID 中的序号排序）
        ids = result["ids"]
        # 排序：按 _ 后的数字
        sorted_pairs = sorted(zip(ids, docs), key=lambda x: int(x[0].split('_')[1]))
        full_text = "\n".join([doc for _, doc in sorted_pairs])
        return {"content": full_text}
    except Exception as e:
        print(f"Error getting file: {e}")
        raise HTTPException(404, "文件未找到或内容为空")

@app.delete("/files")
async def delete_files(request: DeleteRequest):
    """批量删除文件（通过哈希列表），使用 where 条件原子删除"""
    try:
        deleted = rag_engine.delete_files(request.hashes)
        return {"deleted": deleted}
    except Exception as e:
        print(f"Error deleting files: {e}")
        raise HTTPException(status_code=500, detail="删除失败")