import os
from openai import OpenAI
# 这个库是 LangChain 官方提供的文本分割器集合。它的核心任务是将大段文本切分成更小的“块”（Chunks）
from langchain_text_splitters import RecursiveCharacterTextSplitter
# ChromaDB 是一个开源的 AI 原生向量数据库。
# 它的核心作用是存储、管理和查询向量嵌入（Vector Embeddings），这些嵌入是文本、图像等非结构化数据的数值表示
import chromadb
# 一个第三方库（需要 pip install），用于将 .env 文件中的键值对加载到 Python 的环境变量 os.environ 中
from dotenv import load_dotenv

# 1. 加载环境变量
load_dotenv()

# 2. 创建两个客户端
# 2.1 用于 Embedding 的客户端（阿里云百炼）
embedding_client = OpenAI(
    api_key=os.getenv("QIANWEN_API_KEY"),
    base_url=os.getenv("QIANWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
)

# 2.2 用于 Chat 的客户端（DeepSeek）
chat_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL","https://api.deepseek.com")
)

# 3. 初始化 ChromaDB
# 创建持久化客户端。数据会直接写入硬盘的指定目录（./project/chroma_db）。程序重启后数据依然存在，这是生产环境最常用的方式
chroma_client = chromadb.PersistentClient(path="./project/chroma_db")
# chromadb.Client()（无参数）
# 作用：创建内存客户端。数据只存在于当前程序运行期间，关闭即销毁，适合做单元测试或快速原型验证。

# 如果 collection 已存在，先删除，确保从头开始（或者你也可以保留，但建议清空）
try:
    # delete_collection("xxx")	删除整个集合（包括内部所有向量和数据）。不可恢复，物理删除。
    chroma_client.delete_collection("knowledge_base")
except:
    pass
# create_collection(name="xxx")创建一个新集合。如果同名集合已存在且配置（如嵌入函数）不匹配，会报错 UniqueConstraintError
collection = chroma_client.create_collection(name="knowledge_base")

print("程序已启动，准备读取文件...")

# 4. 读取文件内容
file_path = "project1_3/knowledge.txt"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# 5. 切分文本
# RecursiveCharacterTextSplitter 是这个库中最通用、最推荐的文本分割器
# 分割的优先级：段落 (\n\n) > 句子/行 (\n) > 单词 () > 字符 ("")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # 最大允许的字符数
    chunk_overlap=50,  # 相邻两个文本块之间重叠的字符数
    # 对于中文、日文、泰文等书写系统没有空格的语言，默认的分隔符可能导致词语被错误分割。
    # 解决办法是自定义 separators 参数，加入相应的标点符号（如中文的 。、，）作为分隔依据
    separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
)
# split_text()接受原始字符串 content，返回一个字符串列表（List[str]）
chunks = splitter.split_text(content)
print(f"文本已切分为 {len(chunks)} 个片段")

# 6. 批量生成向量
print("正在调用千问 Embedding 模型生成向量...")
response = embedding_client.embeddings.create(
    model="text-embedding-v3",
    input=chunks,          # 批量传入所有 chunk
    # 参数值	    数据类型	                传输体积	        客户端处理成本	                                适用场景
    # "float"	32位浮点数列表 [...]	    较大（明文传输）	    极低（拿来即用）	                        绝大多数本地/开发场景。数据解压即用，无需额外计算。
    # "base64"	压缩编码字符串 "ABc123..."	较小（压缩约 30%~40%）	较高（需 base64.decode + struct.unpack）	高并发生产环境或网络带宽极其有限的场景（如边缘设备）。
    encoding_format="float"
)
# 这是一个 Python 的列表推导式，它的工作是把 API 返回的复杂响应对象，压平成一个纯粹由向量组成的二维列表
embeddings = [item.embedding for item in response.data]
# embeddings = []
# for item in response.data:
    # embeddings.append(item.embedding)
print(f"已生成 {len(embeddings)} 个向量，维度为 {len(embeddings[0])}")

# 7. 存入 ChromaDB
ids = [f"chunk_{i}" for i in range(len(chunks))]
collection.add(
    # 必须，主键。每个文档的唯一身份证。必须唯一，重复添加相同 ID 会报错 UniqueConstraintError
    ids=ids,
    # 可选，原始文本。用于存储和后续展示（如 query 结果中的 documents 字段）
    documents=chunks,
    embeddings=embeddings
)
print("集合中的文档数量：", collection.count())

# 8. 交互式问答循环
print("\n" + "="*50)
print("知识库已准备完成！输入问题开始提问（输入 'exit' 退出）")
print("="*50)

while True:
    user_query = input("\n👤 你：")
    # lower()将用户输入强制转为全小写
    if user_query.lower() in ["exit", "quit"]:
        print("再见！")
        break
    # strip()移除字符串首尾的空白字符（空格、制表符 \t、换行符 \n 等）
    # if not ...：如果剥离空格后字符串变成了空字符串（即用户只敲了回车，或全是空格），则条件成立。
    if not user_query.strip():
        continue

    # 8.1 将用户问题转为向量
    query_embedding = embedding_client.embeddings.create(
        model="text-embedding-v3",
        input=[user_query],
        encoding_format="float"
    ).data[0].embedding  # data（核心向量数据）、usage（Token 消耗统计）、model（实际调用的模型名）

    # 8.2 检索 top-3 相似 chunk
    # collection.query() 负责去向量数据库中抓取与用户问题最相似的文本片段
    # ChromaDB 的 query() 返回的是一个 Python 字典（dict）
    results = collection.query(
        query_embeddings=[query_embedding],  # 传入的向量
        n_results=3  # 返回条数
    )
    # results 是一个字典，包含 ids, documents, distances 等
    retrieved_docs = results['documents'][0]  # 取第一个查询的结果列表

    # 8.3 构造上下文
    # "分隔符".join(列表)：这是 Python 将字符串列表合并为单个字符串的标准方法
    context = "\n\n".join(retrieved_docs)
    # 限制上下文长度，避免超过模型 token 限制（保留足够空间）
    if len(context) > 3000:
        context = context[:3000] + "..."

    # 8.4 构造对话消息
    system_prompt = "你是一个基于给定资料回答问题的助手。请只根据资料内容回答，如果资料中没有相关信息，请说明不知道。"
    user_prompt = f"资料：\n{context}\n\n问题：{user_query}"

    # 8.5 调用 DeepSeek 生成答案（非流式）
    chat_response = chat_client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3,          # 较低温度让回答更稳定
        stream=False
    )
    answer = chat_response.choices[0].message.content
    print(f"\n🤖 AI：{answer}")

    """
    # 8.5 调用 DeepSeek 生成答案(流式)
    chat_response = chat_client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role":"system","content":system_prompt},
            {"role": "user", "content":user_prompt}
        ],
        temperature=0.3,
        stream=True
    )

    # 流式输出
    first_chunk = True  # 第一个片段
    for chunk in chat_response:
        content_piece = chunk.choices[0].delta.content
        if content_piece:
            if first_chuck:
                print(f"\n🤖 AI：",end='',flush=True)  # 只在第一个片段打印一次前缀
                first_chuck=False
            print(content_piece,end='',flush=True)
    # 换行
    print()
    """
