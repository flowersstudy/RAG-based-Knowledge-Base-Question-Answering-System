# RAG 知识库问答系统 Dockerfile
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖（部分Python包需要编译）
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir pdfplumber

# 复制项目代码
COPY main.py rag_engine.py document_processor.py ./
COPY static/ ./static/

# 创建数据目录（用于持久化向量数据库）
RUN mkdir -p /app/chroma_db

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["python", "main.py"]
