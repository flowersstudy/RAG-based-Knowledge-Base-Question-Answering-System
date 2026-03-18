"""
RAG Knowledge Base QA System - FastAPI Backend
"""

import os
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from document_processor import process_uploaded_file
from rag_engine import RAGEngine


def load_env_file(filepath: str = ".env"):
    """加载 .env 文件，兼容 UTF-8 BOM 格式"""
    if not os.path.exists(filepath):
        return

    # 尝试多种编码读取
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']

    for encoding in encodings:
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                for line in f:
                    line = line.strip()
                    # 跳过空行和注释
                    if not line or line.startswith('#'):
                        continue
                    # 解析 KEY=VALUE
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        # 移除可能的引号
                        if value.startswith('"') and value.endswith('"'):
                            value = value[1:-1]
                        elif value.startswith("'") and value.endswith("'"):
                            value = value[1:-1]
                        os.environ[key] = value
            print(f"已加载环境变量: {filepath} (编码: {encoding})")
            return
        except UnicodeDecodeError:
            continue

    print(f"警告: 无法读取 {filepath} 文件")


# 加载环境变量（兼容 Windows UTF-8 BOM）
load_env_file()

# 全局引擎实例
rag_engine: Optional[RAGEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global rag_engine

    # 启动时初始化
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("警告: 未设置 OPENAI_API_KEY 环境变量")
    else:
        base_url = os.getenv("OPENAI_BASE_URL")
        rag_engine = RAGEngine(api_key=api_key, base_url=base_url)
        print("RAG 引擎初始化完成")

    yield

    # 关闭时清理
    print("应用关闭")


app = FastAPI(
    title="RAG Knowledge Base QA System",
    description="基于 RAG 的知识库问答系统",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 数据模型
class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    tokens_used: int


class StatsResponse(BaseModel):
    total_documents: int


# API 路由
@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """获取知识库统计信息"""
    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG 引擎未初始化")

    stats = rag_engine.get_stats()
    return StatsResponse(**stats)


@app.get("/api/documents")
async def get_documents():
    """获取已上传的文档列表"""
    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG 引擎未初始化")

    try:
        documents = rag_engine.get_documents()
        return {"documents": documents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/documents/{filename:path}")
async def delete_document(filename: str):
    """删除指定文档"""
    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG 引擎未初始化")

    try:
        deleted_count = rag_engine.delete_document(filename)
        return {
            "success": True,
            "message": f"已删除文档 '{filename}'",
            "deleted_chunks": deleted_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """上传文档到知识库"""
    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG 引擎未初始化")

    # 检查文件类型
    allowed_extensions = ['.pdf', '.docx', '.doc', '.txt', '.md', '.markdown']
    file_ext = os.path.splitext(file.filename)[1].lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式。支持的格式: {', '.join(allowed_extensions)}"
        )

    try:
        # 读取文件内容
        content = await file.read()

        # 处理文档
        chunks = process_uploaded_file(
            content,
            file.filename,
            chunk_size=500,
            chunk_overlap=50
        )

        # 添加到知识库
        count = rag_engine.add_documents(chunks)

        return JSONResponse({
            "success": True,
            "message": f"成功上传并处理 {file.filename}",
            "chunks_added": count,
            "filename": file.filename
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query")
async def query(request: QueryRequest):
    """问答接口（非流式）"""
    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG 引擎未初始化")

    try:
        result = rag_engine.query(
            question=request.question,
            top_k=request.top_k,
            stream=False
        )

        return QueryResponse(
            answer=result.answer,
            sources=[
                {
                    "content": s.content[:300] + "..." if len(s.content) > 300 else s.content,
                    "source": s.source,
                    "similarity": round(s.similarity, 4)
                }
                for s in result.sources
            ],
            tokens_used=result.tokens_used
        )

    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n{traceback.format_exc()}"
        print(f"错误详情: {error_detail}")
        raise HTTPException(status_code=500, detail=f"{str(e)}\n请检查网络连接和API配置")


@app.post("/api/query/stream")
async def query_stream(request: QueryRequest):
    """问答接口（流式输出）"""
    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG 引擎未初始化")

    try:
        generator, sources = rag_engine.query(
            question=request.question,
            top_k=request.top_k,
            stream=True
        )

        # 构建 SSE 流
        async def event_generator():
            # 发送来源信息
            sources_data = [
                {
                    "content": s.content[:300] + "..." if len(s.content) > 300 else s.content,
                    "source": s.source,
                    "similarity": round(s.similarity, 4)
                }
                for s in sources
            ]
            yield f"data: {json.dumps({'type': 'sources', 'data': sources_data})}\n\n"

            # 发送回答内容
            for chunk in generator:
                yield f"data: {json.dumps({'type': 'content', 'data': chunk})}\n\n"

            # 发送结束标记
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/clear")
async def clear_knowledge_base():
    """清空知识库"""
    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG 引擎未初始化")

    try:
        rag_engine.clear_all()
        return {"success": True, "message": "知识库已清空"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 静态文件服务
try:
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    os.makedirs(static_dir, exist_ok=True)
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
except Exception as e:
    print(f"静态文件目录配置失败: {e}")


@app.get("/api/health")
async def health_check():
    """健康检查"""
    return {
        "status": "ok",
        "rag_engine_ready": rag_engine is not None
    }


if __name__ == "__main__":
    import uvicorn
    import json  # for stream endpoint

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))

    print(f"启动服务: http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
