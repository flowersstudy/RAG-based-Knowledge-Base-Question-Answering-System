# RAG-based Knowledge Base Question Answering System

基于 RAG (Retrieval-Augmented Generation) 的知识库问答系统，支持多种文档格式的上传、智能检索和问答。

## 🚀 在线体验

**Render 部署版（无需安装，直接用）：**
> https://rag-knowledge-base.onrender.com

> ⚠️ 注意：Render 免费版会在 15 分钟无访问后休眠，首次访问可能需要等待 30 秒启动。

**自己部署：** 点击下方按钮一键部署到你的 Render 账号

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/flowersstudy/RAG-based-Knowledge-Base-Question-Answering-System)

## 功能特性

- 📄 **多格式文档支持**：PDF、Word、TXT、Markdown
- 🔍 **智能检索**：基于向量相似度的语义检索
- 🤖 **大模型问答**：支持 Kimi (Moonshot) / OpenAI API
- 🗄️ **向量数据库**：使用 ChromaDB 存储文档向量
- 🌐 **Web 界面**：简洁美观的交互界面
- ⚡ **流式输出**：实时显示回答内容

## 技术栈

- **后端**：Python + FastAPI
- **向量数据库**：ChromaDB
- **Embedding**：本地 sentence-transformers 模型（支持中文）
- **LLM**：Kimi (Moonshot) / OpenAI API
- **前端**：HTML + JavaScript + Tailwind CSS
- **文档处理**：PyPDF2、python-docx

## 快速开始

### 本地直接运行（Python + pip）

**1. 克隆项目并进入目录**
```bash
git clone https://github.com/flowersstudy/RAG-based-Knowledge-Base-Question-Answering-System.git
cd RAG-based-Knowledge-Base-Question-Answering-System
```

**2. 安装依赖**
```bash
pip install -r requirements.txt
```

**3. 配置 API 密钥**

创建 `.env` 文件（参考 `.env.example`）：

**使用 Kimi (推荐)：**
```env
OPENAI_API_KEY=sk-your-kimi-api-key
OPENAI_BASE_URL=https://api.moonshot.cn/v1
LLM_MODEL=moonshot-v1-8k
```

**使用 阿里云通义千问：**
```env
OPENAI_API_KEY=sk-your-aliyun-api-key
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen-turbo
```

**4. 启动服务**
```bash
python main.py
```

**5. 访问系统**
打开浏览器访问 http://localhost:8000

---

## 项目结构

```
.
├── main.py              # FastAPI 主程序
├── rag_engine.py        # RAG 核心引擎
├── document_processor.py # 文档处理模块
├── requirements.txt     # 依赖列表
├── .env.example         # 环境变量示例
├── static/              # 静态文件
│   └── index.html       # 前端页面
└── chroma_db/           # 向量数据库目录
```

## API 文档

启动服务后访问：http://localhost:8000/docs

## 界面预览

![本地图片](./images/UI%20Example%20.jpeg)

### 主要功能区域
- **左侧**：文档上传区 + 已上传文档列表 + 删除按钮
- **右侧**：问答对话框 + 流式输出显示

## License

MIT License
