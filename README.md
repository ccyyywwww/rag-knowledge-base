# PDF 知识库问答系统

一个基于 RAG（Retrieval-Augmented Generation，检索增强生成）的文档问答系统。支持**批量上传 PDF/TXT**、**增量建库**、**文件管理**，上传后可向 AI 提问文档中的内容。

## 功能

- 批量上传多个 PDF / TXT 文件，自动构建向量知识库
- **增量更新**：通过文件哈希（SHA256）去重，只处理新增/修改的文件，不重复入库
- 基于文档内容回答问题，AI 只依据检索到的资料作答，不会凭空编造
- **文件管理**：列出已入库文件、查看文件原文、删除文件（向量同步删除）
- 支持中文文档
- Web 界面操作，无需命令行
- 上传安全防护：防路径穿越、扩展名白名单、50MB 文件大小限制

## 技术栈

- Python 3.10+
- FastAPI（Web 接口）+ Uvicorn（服务器）
- ChromaDB（向量数据库）
- LangChain `RecursiveCharacterTextSplitter`（文本切分）
- 千问 `text-embedding-v3`（Embedding 模型）
- DeepSeek API（问答生成）

## 项目结构

```
├── app.py            # FastAPI 接口层：批量上传、提问、文件管理
├── rag_engine.py     # RAG 核心逻辑：哈希索引、增量入库、检索、生成、删除
├── static/index.html # 前端页面
├── requirements.txt  # 依赖清单
├── .env.example      # 环境变量模板
└── .env              # API 密钥（已被 .gitignore 忽略，不入库）
```

> `file_index.json`（文件哈希索引）和 `chroma_db/`（向量库）、`uploads/`（临时上传）都已加入 `.gitignore`，不会提交到仓库。

## 如何运行

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

国内网络慢的话，用清华镜像：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. 配置环境变量

在项目根目录创建 `.env` 文件（参考 `.env.example`）：

```bash
# DeepSeek API（负责最终回答）
DEEPSEEK_API_KEY=你的DeepSeek密钥
DEEPSEEK_BASE_URL=https://api.deepseek.com

# 阿里云千问 API（负责文本向量化，因为 DeepSeek 没有 embedding 接口）
QIANWEN_API_KEY=你的千问密钥
QIANWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

> DeepSeek 密钥在 [platform.deepseek.com](https://platform.deepseek.com) 创建
> 千问密钥在 [阿里云百炼](https://bailian.console.aliyun.com) 创建

### 3. 启动服务

**必须在项目目录下运行**（否则报 `Could not import module "app"`）：

```bash
cd 项目目录
uvicorn app:app --reload
```

不想切目录的话，用 `--app-dir` 指定项目路径，从任何位置都能启动：

```bash
uvicorn app:app --reload --app-dir "D:\桌面\就业\project1_3"
```

### 4. 使用

浏览器打开 <http://127.0.0.1:8000>，批量上传文档 → 建库 → 提问。页面上可直接查看/删除已入库的文件。

## 接口说明

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 返回前端页面 |
| POST | `/upload` | 批量上传文档（multipart/form-data，`files` 字段），增量建库 |
| POST | `/ask` | 提问（JSON `{"question": "..."}`），返回 AI 回答 |
| GET | `/files` | 列出所有已入库文件（哈希 + 原始名） |
| GET | `/file/{hash}` | 查看指定文件的完整原文 |
| DELETE | `/files` | 批量删除文件（JSON `{"hashes": [...]}`），向量同步删除 |

## 更新日志

### v2.0（增量更新 + 文件管理）
- 支持**批量上传**多个文件
- **增量建库**：用 SHA256 哈希判断文件是否已处理，新增/修改才入库
- **metadata 存储**：每个文本片段附带 `file_hash`，支持按文件精确查询/删除
- 新增文件管理接口：`/files`、`/file/{hash}`、`DELETE /files`
- 空知识库提问时返回友好提示，不再抛英文异常
- 向量化分批处理（每批 10 段），避免 API 限流

### v2.1（安全加固）
- 补回上传接口的**扩展名白名单**（仅 .pdf/.txt）
- 补回**50MB 文件大小限制**（流式写入时按字节计数，超限即中止并清理）
- 将 `file_index.json` 加入 `.gitignore`，避免本地索引误提交

## 已知不足

- **增量更新只增不减**：文件内容修改后重新入库，但旧版本的向量不会自动清理，可能积累重复片段（当前通过"删除旧文件再上传"规避）
- **上传文件在入库后即删除**：`uploads/` 只做临时中转，原文以文本形式存于 ChromaDB
- **无并发保护**：多个请求同时上传时可能竞争同一集合，学习项目阶段未加锁
- 扫描版 / 图片型 PDF 没有文字层，`pypdf` 无法提取文本，建议使用带文字的 PDF

## 说明

- `uploads/`、`chroma_db/`、`file_index.json`、`.env` 已被 `.gitignore` 忽略，不会提交到仓库
