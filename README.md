# PDF 知识库问答系统

一个基于 RAG（Retrieval-Augmented Generation，检索增强生成）的文档问答系统。上传 PDF 或 TXT 文档，即可向 AI 提问文档中的内容。

## 功能

- 上传 PDF / TXT 文件，自动构建向量知识库
- 基于文档内容回答问题，AI 只依据检索到的资料作答，不会凭空编造
- 支持中文文档
- Web 界面操作，无需命令行
- 上传安全防护：防路径穿越、文件大小限制、扩展名白名单

## 技术栈

- Python 3.10+
- FastAPI（Web 接口）+ Uvicorn（服务器）
- ChromaDB（向量数据库）
- LangChain `RecursiveCharacterTextSplitter`（文本切分）
- 千问 `text-embedding-v3`（Embedding 模型）
- DeepSeek API（问答生成）

## 项目结构

```
├── app.py            # FastAPI 接口层：/upload 上传建库，/ask 提问
├── rag_engine.py     # RAG 核心逻辑：文档解析、切分、向量化、检索、生成
├── static/index.html # 前端页面
├── requirements.txt  # 依赖清单
└── .env              # API 密钥（已被 .gitignore 忽略，不入库）
```

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

```bash
cd 项目目录
uvicorn app:app --reload
```

### 4. 使用

浏览器打开 <http://127.0.0.1:8000>，先上传文档，再提问。

## 接口说明

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 返回前端页面 |
| POST | `/upload` | 上传文档（multipart/form-data），解析后构建知识库 |
| POST | `/ask` | 提问（JSON `{"question": "..."}`），返回 AI 回答 |

## 说明

- 扫描版 / 图片型 PDF 没有文字层，`pypdf` 无法提取文本，建议使用带文字的 PDF
- 每次上传文档都会重建整个知识库（学习项目阶段的设计）
- `uploads/`、`chroma_db/`、`.env` 已被 `.gitignore` 忽略，不会提交到仓库
