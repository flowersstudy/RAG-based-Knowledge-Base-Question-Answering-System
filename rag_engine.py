"""
RAG 核心引擎
实现文档向量化存储和检索问答功能
支持 Kimi (Moonshot) API
"""

import os
from typing import List, Dict, Optional
from dataclasses import dataclass
import hashlib

import chromadb
from chromadb.config import Settings
from openai import OpenAI

from document_processor import DocumentChunk


class LocalEmbedding:
    """本地 Embedding 模型（基于文本特征，无需下载）"""

    def __init__(self):
        print("使用文本特征向量化方案（无需下载模型）")

    def encode(self, texts: List[str]) -> List[List[float]]:
        """编码文本为向量"""
        return self._text_feature_encode(texts)

    def _text_feature_encode(self, texts: List[str]) -> List[List[float]]:
        """基于文本特征的向量化"""
        import math
        from collections import Counter

        embeddings = []
        common_chars = ['的', '是', '在', '有', '和', '了', '不', '人', '我', '一',
                       'a', 'e', 'i', 'o', 'n', 'r', 't', 's', 'l', 'c']

        for text in texts:
            features = []
            text_len = max(len(text), 1)

            # 1. 基础统计特征
            features.append(len(text) / 1000.0)
            features.append(len(set(text)) / text_len)

            # 2. 字符类型比例
            chinese = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
            english = sum(1 for c in text if c.isalpha() and c.isascii())
            digits = sum(1 for c in text if c.isdigit())
            spaces = sum(1 for c in text if c.isspace())
            features.extend([chinese/text_len, english/text_len, digits/text_len, spaces/text_len])

            # 3. 常见字符频率
            char_freq = Counter(text.lower())
            for char in common_chars:
                features.append(char_freq.get(char, 0) / text_len)

            # 4. 哈希特征（增加区分度）
            h = hash(text)
            for i in range(50):
                features.append(((h >> (i % 32)) & 0xFF) / 255.0)

            # 填充到384维
            while len(features) < 384:
                features.append(0.0)
            features = features[:384]

            # 归一化
            norm = math.sqrt(sum(x*x for x in features))
            if norm > 0:
                features = [x/norm for x in features]
            embeddings.append(features)

        return embeddings


@dataclass
class SearchResult:
    """搜索结果"""
    content: str
    source: str
    similarity: float


@dataclass
class Answer:
    """回答结果"""
    answer: str
    sources: List[SearchResult]
    tokens_used: int = 0


class RAGEngine:
    """RAG 引擎 - 支持 Kimi API"""

    def __init__(self, api_key: str, base_url: Optional[str] = None):
        self.api_key = api_key
        # 默认使用 Kimi API
        self.base_url = base_url or "https://api.moonshot.cn/v1"

        # 获取模型配置
        self.llm_model = os.getenv("LLM_MODEL", "moonshot-v1-8k")

        # 初始化 OpenAI 客户端（Kimi 兼容 OpenAI 格式）
        self.client = OpenAI(api_key=api_key, base_url=self.base_url)

        # 初始化本地 Embedding 模型（Kimi 不提供 embedding）
        self.embedding_model = LocalEmbedding()

        # 初始化 ChromaDB
        db_path = os.path.join(os.path.dirname(__file__), "chroma_db")
        os.makedirs(db_path, exist_ok=True)

        self.chroma_client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )

        # 获取或创建集合
        self.collection = self.chroma_client.get_or_create_collection(
            name="knowledge_base",
            metadata={"hnsw:space": "cosine"}
        )

    def _get_embedding(self, text: str) -> List[float]:
        """获取文本的 embedding（使用本地模型）"""
        embeddings = self.embedding_model.encode([text[:8000]])
        return embeddings[0]

    def add_documents(self, chunks: List[DocumentChunk]) -> int:
        """添加文档到知识库"""
        if not chunks:
            return 0

        # 准备数据
        ids = []
        documents = []
        metadatas = []
        embeddings = []

        for chunk in chunks:
            # 生成唯一 ID
            doc_id = hashlib.md5(
                f"{chunk.source}:{chunk.chunk_id}:{chunk.content[:100]}".encode()
            ).hexdigest()

            ids.append(doc_id)
            documents.append(chunk.content)
            metadatas.append({
                "source": chunk.source,
                "chunk_id": chunk.chunk_id
            })

        # 获取 embeddings
        print(f"正在生成 {len(chunks)} 个文档块的 embedding...")
        for doc in documents:
            embedding = self._get_embedding(doc)
            embeddings.append(embedding)

        # 添加到 ChromaDB
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )

        return len(chunks)

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """搜索相关文档"""
        # 获取查询的 embedding
        query_embedding = self._get_embedding(query)

        # 搜索
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        # 解析结果
        search_results = []
        if results['documents'] and len(results['documents'][0]) > 0:
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                distance = results['distances'][0][i] if results['distances'] else 0

                # 将距离转换为相似度 (cosine distance -> similarity)
                similarity = 1 - distance

                search_results.append(SearchResult(
                    content=doc,
                    source=metadata.get('source', 'Unknown'),
                    similarity=similarity
                ))

        return search_results

    def query(self, question: str, top_k: int = 5, stream: bool = False):
        """
        问答接口
        返回 Answer 对象或生成器（流式输出）
        """
        # 检索相关文档
        search_results = self.search(question, top_k)

        if not search_results:
            if stream:
                def empty_generator():
                    yield "抱歉，知识库中没有找到相关信息。"
                return empty_generator()
            return Answer(
                answer="抱歉，知识库中没有找到相关信息。",
                sources=[]
            )

        # 构建上下文
        context_parts = []
        for i, result in enumerate(search_results):
            context_parts.append(f"[文档{i+1}] 来源: {result.source}\n{result.content}")
        context = "\n\n".join(context_parts)

        # 构建提示词
        system_prompt = """你是一个专业的知识库问答助手。请基于提供的参考资料回答用户的问题。

重要规则：
1. 只能基于提供的参考资料回答，不要编造信息
2. 如果参考资料不足以回答问题，请明确说明
3. 回答要准确、简洁、有条理
4. 引用相关文档时可以使用 [文档X] 的格式"""

        user_prompt = f"""参考资料：
{context}

用户问题：{question}

请基于以上参考资料回答问题。"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        if stream:
            # 流式输出
            def response_generator():
                response = self.client.chat.completions.create(
                    model=self.llm_model,
                    messages=messages,
                    stream=True,
                    temperature=0.7
                )

                for chunk in response:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content

            return response_generator(), search_results
        else:
            # 非流式输出
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=0.7
            )

            answer_text = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else 0

            return Answer(
                answer=answer_text,
                sources=search_results,
                tokens_used=tokens_used
            )

    def get_stats(self) -> Dict:
        """获取知识库统计信息"""
        count = self.collection.count()
        return {
            "total_documents": count
        }

    def clear_all(self):
        """清空知识库"""
        self.chroma_client.delete_collection("knowledge_base")
        self.collection = self.chroma_client.create_collection(
            name="knowledge_base",
            metadata={"hnsw:space": "cosine"}
        )
