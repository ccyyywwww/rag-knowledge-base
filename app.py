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
# UUID = Universally Unique IDentifier（通用唯一标识符）。uuid4() 是其中一种生成算法（基于随机数）
from uuid import uuid4
# UploadFile：它是一个类型。
# 当你在函数参数里写 file: UploadFile = File(...) 时，FastAPI 会把接收到的文件包装成一个 UploadFile 对象交给你
from fastapi import FastAPI, UploadFile, File, HTTPException
# FastAPI 提供的静态文件挂载器
from fastapi.staticfiles import StaticFiles
# FileResponse：用来直接把服务器上的某个文件作为响应内容发送给客户端（比如点一个链接就自动下载 PDF）
# JSONResponse：专门用来返回 JSON 格式数据的响应对象
from fastapi.responses import FileResponse, JSONResponse
# Pydantic 是 Python 最流行的数据校验库。BaseModel 是它的基类。
from pydantic import BaseModel

import rag_engine

# 创建 FastAPI 应用实例。title 只是文档里显示的名字，不影响功能
app = FastAPI(title="PDF 知识库问答系统")

# 把 static 目录挂载为静态资源，浏览器就能访问里面的 index.html
app.mount("/static", StaticFiles(directory=os.path.join(rag_engine.BASE_DIR, "static")), name="static")

# 确保上传目录存在（第一次运行时会自动创建）
os.makedirs(rag_engine.UPLOAD_DIR, exist_ok=True)


# ---------- 首页 ----------
# 告诉 FastAPI：“如果用户发来一个 GET 请求，并且访问的路径是根路径 /（即域名后面什么都没带），就执行下面这个函数。”
@app.get("/")
async def index():
    # 这不是注释，而是函数的 __doc__。FastAPI 会自动把它提取到自动生成的交互文档（/docs）里，方便团队协作
    """把前端页面返回给浏览器。"""
    return FileResponse(os.path.join(rag_engine.BASE_DIR, "static", "index.html"))


# ---------- 上传接口 ----------
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """
    接收上传的文档（PDF 或 TXT），解析入库。

    UploadFile 是 FastAPI 对"上传文件"的封装。
    它读取的流默认存在内存/临时文件里，必须先存到磁盘，rag_engine 才能读取。

    安全措施：
    1. basename + 扩展名白名单 + uuid 重命名 → 防路径穿越 / 同名覆盖
    2. 边读边计数，超过 50MB 中止 → 防超大文件写爆磁盘
    """
    # ① 防路径穿越 + 防同名覆盖
    # 不能直接用 file.filename：攻击者可传 "../../etc/passwd"，
    # os.path.join 会拼出 uploads/../../etc/passwd，跳出上传目录。
    filename = os.path.basename(file.filename)   # 去掉所有路径，只剩文件名
    ext = os.path.splitext(filename)[1].lower()  # 取扩展名，如 ".pdf"

    # 扩展名白名单。前端的 accept=".pdf,.txt" 只是给用户的提示，不是安全边界，
    # 后端必须自己再校验一遍（攻击者可以不经过浏览器直接发请求）。
    if ext not in {".pdf", ".txt"}:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型：{ext}")

    # uuid 重命名：uuid4() 生成随机 32 位十六进制串，.hex 去掉连字符。
    # 好处：同名文件不会互相覆盖，且隐藏了真实文件名（防扩展名伪装）。
    safe_name = f"{uuid4().hex}{ext}"
    file_path = os.path.join(rag_engine.UPLOAD_DIR, safe_name)

    # ② 防超大文件
    # 为什么不用 Content-Length 预判？两个原因：
    #   1. 它是 HTTP 头，可以被攻击者伪造；
    #   2. 分块传输（chunked）的请求根本没有这个头。
    # 所以可靠做法是读取数据流时按字节累计，超限立即中止。
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

    size = 0
    try:
        with open(file_path, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)  # 每次读 1MB。注意 await：异步 I/O
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_FILE_SIZE:
                    raise HTTPException(status_code=413, detail="文件超过 50MB 限制")
                f.write(chunk)
    except HTTPException:
        # 超限时删除已写入的残留文件，别在磁盘上留垃圾
        if os.path.exists(file_path):
            os.remove(file_path)
        raise

    # ③ 调用 RAG 引擎建库
    try:
        chunk_count = rag_engine.rebuild_knowledge_base(file_path)
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": f"建库失败: {e}"})

    return {"message": f"上传成功，已切分为 {chunk_count} 个片段", "chunk_count": chunk_count}


# ---------- 提问接口 ----------
class AskRequest(BaseModel):
    """请求体。FastAPI 会自动校验 JSON 里必须有 question 字段。"""
    question: str


@app.post("/ask")
async def ask(req: AskRequest):
    """接收 {'question': '...'}，返回 {'answer': ..., 'sources': [...]}。"""
    # 先检查知识库是否已建立。没上传过文档就提问，会让用户看到一串英文报错，
    # 不如直接给一句中文提示。
    if not rag_engine.has_knowledge_base():
        return JSONResponse(status_code=400, content={"detail": "请先上传文档并建库，再开始提问"})

    try:
        result = rag_engine.ask(req.question)
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": f"提问失败: {e}"})
    return result
