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

### 方式一：Docker 一键运行（本地部署）

**1. 克隆项目并进入目录**
```bash
git clone https://github.com/flowersstudy/RAG-based-Knowledge-Base-Question-Answering-System.git
cd RAG-based-Knowledge-Base-Question-Answering-System
```

**2. 配置 API 密钥**
```bash
# 复制配置文件
cp .env.example .env

# 编辑 .env 文件，填入你的 API 密钥
# Windows: notepad .env
# Mac/Linux: nano .env
```

**3. 启动服务**
```bash
docker-compose up -d
```

**4. 访问系统**
打开浏览器访问 http://localhost:8000

**常用命令**
```bash
# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down

# 重启服务
docker-compose restart

# 更新镜像（代码有更新时）
docker-compose up -d --build
```

---

### 方式二：部署到 Render（推荐，免费在线访问）

**1. 点击部署按钮**

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/flowersstudy/RAG-based-Knowledge-Base-Question-Answering-System)

**2. 配置环境变量**

在 Render 控制台设置以下环境变量：
- `OPENAI_API_KEY`：你的 API 密钥（必填）
- `OPENAI_BASE_URL`：API 基础地址（如 `https://api.moonshot.cn/v1`）
- `LLM_MODEL`：模型名称（如 `moonshot-v1-8k`）

**3. 等待部署完成**

大约 2-3 分钟后，Render 会给你一个访问链接，直接打开就能用！

> ⚠️ 注意：Render 免费版会在 15 分钟无访问后休眠，首次访问可能需要等待 30 秒启动。

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
