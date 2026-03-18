"""
Hugging Face Spaces 版本的 RAG 应用
使用 Gradio 界面
"""

import os
import tempfile
import gradio as gr
from rag_engine import RAGEngine, Answer
from document_processor import process_uploaded_file

# 加载环境变量
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

# 初始化 RAG 引擎
rag_engine = None
if api_key:
    try:
        rag_engine = RAGEngine(api_key=api_key, base_url=base_url)
        print("RAG 引擎初始化完成")
    except Exception as e:
        print(f"RAG 引擎初始化失败: {e}")
else:
    print("警告: 未设置 OPENAI_API_KEY")


def upload_file(file):
    """上传文档"""
    if not rag_engine:
        return "❌ 错误: RAG 引擎未初始化，请检查 API Key 配置"

    if file is None:
        return "请先选择文件"

    try:
        # 读取文件内容
        with open(file.name, 'rb') as f:
            content = f.read()

        filename = os.path.basename(file.name)

        # 处理文档
        chunks = process_uploaded_file(content, filename, chunk_size=500, chunk_overlap=50)

        # 添加到知识库
        count = rag_engine.add_documents(chunks)

        return f"✅ 成功上传 '{filename}'，已处理 {count} 个片段"
    except Exception as e:
        return f"❌ 上传失败: {str(e)}"


def query_documents(question):
    """查询文档"""
    if not rag_engine:
        return "❌ 错误: RAG 引擎未初始化，请检查 API Key 配置"

    if not question.strip():
        return "请输入问题"

    try:
        result = rag_engine.query(question=question, top_k=5, stream=False)

        answer = result.answer

        # 添加来源信息
        if result.sources:
            answer += "\n\n📚 **参考来源:**\n"
            for i, source in enumerate(result.sources[:3], 1):
                similarity = source.similarity * 100
                content = source.content[:200] + "..." if len(source.content) > 200 else source.content
                answer += f"\n[{i}] {source.source} (相似度: {similarity:.1f}%)\n{content}\n"

        return answer
    except Exception as e:
        return f"❌ 查询失败: {str(e)}"


def get_doc_stats():
    """获取文档统计"""
    if not rag_engine:
        return "RAG 引擎未初始化"

    try:
        stats = rag_engine.get_stats()
        docs = rag_engine.get_documents()

        result = f"📊 **知识库统计**\n\n"
        result += f"总片段数: {stats.get('total_documents', 0)}\n\n"

        if docs:
            result += "**已上传文档:**\n"
            for doc in docs:
                result += f"- {doc['filename']}: {doc['chunks']} 片段\n"
        else:
            result += "暂无文档\n"

        return result
    except Exception as e:
        return f"获取统计失败: {str(e)}"


# 创建 Gradio 界面
with gr.Blocks(title="RAG 知识库问答系统", css="""
    .container { max-width: 1200px; margin: 0 auto; }
    .header { text-align: center; margin-bottom: 20px; }
    .header h1 { color: #667eea; }
""") as demo:

    gr.HTML("""
    <div class="header">
        <h1>🧠 RAG 知识库问答系统</h1>
        <p>上传文档，智能问答</p>
    </div>
    """)

    with gr.Row():
        # 左侧：上传和统计
        with gr.Column(scale=1):
            gr.Markdown("### 📁 文档上传")
            file_input = gr.File(
                label="选择文件 (PDF, Word, TXT, MD)",
                file_types=[".pdf", ".docx", ".doc", ".txt", ".md", ".markdown"]
            )
            upload_btn = gr.Button("上传文档", variant="primary")
            upload_result = gr.Textbox(label="上传结果", lines=2)

            gr.Markdown("---")

            gr.Markdown("### 📊 知识库状态")
            stats_btn = gr.Button("刷新统计")
            stats_output = gr.Textbox(label="统计信息", lines=10)

        # 右侧：问答
        with gr.Column(scale=2):
            gr.Markdown("### 💬 智能问答")
            question_input = gr.Textbox(
                label="输入你的问题",
                placeholder="例如：这份文档的主要内容是什么？",
                lines=2
            )
            query_btn = gr.Button("提问", variant="primary")
            answer_output = gr.Textbox(label="回答", lines=15)

    # 绑定事件
    upload_btn.click(upload_file, inputs=file_input, outputs=upload_result)
    query_btn.click(query_documents, inputs=question_input, outputs=answer_output)
    stats_btn.click(get_doc_stats, outputs=stats_output)

    # 页面加载时刷新统计
    demo.load(get_doc_stats, outputs=stats_output)


if __name__ == "__main__":
    demo.launch()
