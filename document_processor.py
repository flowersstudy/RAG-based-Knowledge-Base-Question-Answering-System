"""
文档处理模块
支持 PDF、Word、TXT、Markdown 等格式的文档解析
"""

import os
import re
from typing import List
from dataclasses import dataclass
import tempfile


@dataclass
class DocumentChunk:
    """文档分片"""
    content: str
    source: str
    page_num: int = 0
    chunk_id: int = 0


class DocumentProcessor:
    """文档处理器"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def process_file(self, file_path: str, filename: str) -> List[DocumentChunk]:
        """处理文件并返回文本分片"""
        ext = os.path.splitext(filename)[1].lower()

        # 读取文件内容
        if ext == '.pdf':
            text = self._read_pdf(file_path)
        elif ext in ['.docx', '.doc']:
            text = self._read_word(file_path)
        elif ext in ['.txt', '.md', '.markdown']:
            text = self._read_text(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {ext}")

        # 文本清洗
        text = self._clean_text(text)

        # 分片
        chunks = self._split_text(text, filename)

        return chunks

    def _read_pdf(self, file_path: str) -> str:
        """读取 PDF 文件"""
        try:
            import PyPDF2
            text_parts = []
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text:
                        text_parts.append(f"[第{page_num + 1}页]\n{text}")
            return '\n\n'.join(text_parts)
        except Exception as e:
            raise Exception(f"PDF 读取失败: {str(e)}")

    def _read_word(self, file_path: str) -> str:
        """读取 Word 文件"""
        try:
            from docx import Document
            doc = Document(file_path)
            paragraphs = []
            for para in doc.paragraphs:
                if para.text.strip():
                    paragraphs.append(para.text)
            return '\n'.join(paragraphs)
        except Exception as e:
            raise Exception(f"Word 读取失败: {str(e)}")

    def _read_text(self, file_path: str) -> str:
        """读取文本文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # 尝试其他编码
            with open(file_path, 'r', encoding='gbk') as f:
                return f.read()

    def _clean_text(self, text: str) -> str:
        """清洗文本"""
        # 移除多余空白
        text = re.sub(r'\s+', ' ', text)
        # 移除特殊字符
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f]', '', text)
        # 规范化换行
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        return text.strip()

    def _split_text(self, text: str, source: str) -> List[DocumentChunk]:
        """将文本分片"""
        chunks = []

        # 按段落分割
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]

        current_chunk = ""
        chunk_id = 0

        for para in paragraphs:
            # 如果当前段落过长，需要进一步分割
            if len(para) > self.chunk_size:
                # 先保存当前累积的内容
                if current_chunk:
                    chunks.append(DocumentChunk(
                        content=current_chunk.strip(),
                        source=source,
                        chunk_id=chunk_id
                    ))
                    chunk_id += 1
                    current_chunk = ""

                # 分割长段落
                for i in range(0, len(para), self.chunk_size - self.chunk_overlap):
                    sub_chunk = para[i:i + self.chunk_size]
                    chunks.append(DocumentChunk(
                        content=sub_chunk,
                        source=source,
                        chunk_id=chunk_id
                    ))
                    chunk_id += 1

            else:
                # 检查加入当前段落后是否超过限制
                if len(current_chunk) + len(para) > self.chunk_size:
                    # 保存当前分片
                    if current_chunk:
                        chunks.append(DocumentChunk(
                            content=current_chunk.strip(),
                            source=source,
                            chunk_id=chunk_id
                        ))
                        chunk_id += 1

                    # 保留部分重叠文本
                    if len(current_chunk) > self.chunk_overlap:
                        words = current_chunk.split()
                        overlap_text = ' '.join(words[-self.chunk_overlap // 5:])
                        current_chunk = overlap_text + "\n" + para
                    else:
                        current_chunk = para
                else:
                    if current_chunk:
                        current_chunk += "\n" + para
                    else:
                        current_chunk = para

        # 保存最后一个分片
        if current_chunk:
            chunks.append(DocumentChunk(
                content=current_chunk.strip(),
                source=source,
                chunk_id=chunk_id
            ))

        return chunks


def process_uploaded_file(file_content: bytes, filename: str,
                          chunk_size: int = 500,
                          chunk_overlap: int = 50) -> List[DocumentChunk]:
    """处理上传的文件内容"""
    # 创建临时文件
    suffix = os.path.splitext(filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_content)
        tmp_path = tmp.name

    try:
        processor = DocumentProcessor(chunk_size, chunk_overlap)
        return processor.process_file(tmp_path, filename)
    finally:
        # 清理临时文件
        os.unlink(tmp_path)
