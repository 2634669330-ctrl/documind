<div align="center">

# DocuMind

**基于 RAG 的企业级智能文档知识库问答系统**

把一个文件夹变成可以对话的知识库 —— 上传 PDF / Word / Markdown，用自然语言提问，秒级拿到**带原文出处的答案**。

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.112-009688?logo=fastapi&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.2-1C3C3C)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-FF6B6B)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-4169E1?logo=postgresql&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

[功能特性](#-核心特性) · [系统架构](#-系统架构) · [快速开始](#-快速开始) · [API](#-api-文档) · [关键实现](#-关键实现) · [效果评测](#-效果与评测) · [Roadmap](#-roadmap)

</div>

---

## 📌 项目简介

企业内部的知识大多沉淀在 PDF、Word、Markdown 这类**非结构化长文档**里：产品手册、技术规范、投标文件、会议纪要。它们有三个共同的痛点：

| 痛点 | 具体表现 |
| --- | --- |
| **找不到** | 关键词搜索只能命中字面，问「出差住宿标准是多少」搜不到写着「差旅费报销细则」的章节 |
| **读不完** | 一份 200 页的规范，人工定位一个条款要翻十几分钟 |
| **喂不进模型** | 文档远超上下文窗口，且涉及内部资料，不适合直接上传公有云 |

DocuMind 用 **RAG（Retrieval-Augmented Generation，检索增强生成）** 解决这件事，把回答建立在「先检索、再生成」之上：

- **离线索引**：文档解析 → 语义分块 → 向量化 → 写入向量库
- **在线问答**：问题向量化 → 混合检索 → 重排 → 拼装 Prompt → 大模型流式生成 → **答案附带原文页码**

这样做的好处是：模型不需要「记住」文档，只需要「读懂」检索出来的几段原文，因此**幻觉可控、答案可溯源、文档可随时增删**。

> 设计原则：**宁可回答「文档里没有」，也不编造。** 所有涉及事实的句子都必须能点回原文。

---

## ✨ 核心特性

| 能力 | 说明 |
| --- | --- |
| 📄 多格式解析 | PDF / DOCX / TXT / Markdown，保留页码、章节标题等元数据 |
| ✂️ 语义分块 | `RecursiveCharacterTextSplitter`，chunk 512 token、overlap 64，按段落 / 标题递归切分，避免把一句话切断 |
| 🧠 本地向量化 | BGE-m3 生成 1024 维向量，Embedding 全程本地推理，**文档内容不出内网** |
| 🗂️ 向量库持久化 | ChromaDB 持久化存储，支持按文档 ID 增量入库与删除，无需重建全量索引 |
| 🔍 双路召回 | 向量语义召回 + BM25 关键词召回，互补「同义不同词」与「专有名词精确匹配」两类场景 |
| 🎯 重排（Rerank） | `bge-reranker-v2-m3` 对 Top-20 候选二次打分，取 Top-5 送入模型，显著降低答非所问 |
| 💬 多轮对话 | LangChain Memory 保存对话历史，支持「它」「这个功能」这类指代追问 |
| ⚡ 流式输出 | SSE 推流，首 token 约 1 秒可见，边生成边展示 |
| 📎 引用溯源 | 每个结论标注来源编号，前端可展开查看原文片段与页码 |
| 🛡️ 防幻觉 | 上下文无依据时明确拒答，Prompt 层强制约束输出结构 |
| 🔐 会话持久化 | PostgreSQL + SQLAlchemy 存会话、消息、引用，刷新页面历史不丢 |
| 🐳 一键部署 | 多阶段构建 Dockerfile + docker-compose，Web / PostgreSQL / 向量库三条命令起服务 |

---

## 🏗 系统架构

```mermaid
flowchart LR
    subgraph offline["离线索引"]
        A["📄 上传文档<br/>PDF / DOCX / MD / TXT"] --> B["解析 + 元数据抽取<br/>pypdf / python-docx"]
        B --> C["语义分块<br/>512 token / overlap 64"]
        C --> D["BGE-m3 向量化<br/>1024-dim"]
        D --> E[("ChromaDB<br/>向量 + 元数据")]
    end

    subgraph online["在线问答"]
        Q["🙋 用户提问"] --> R["问题向量化"]
        R --> S["向量召回 Top-20"]
        R --> S2["BM25 召回 Top-20"]
        S --> T["合并去重"]
        S2 --> T
        T --> U["bge-reranker 重排<br/>取 Top-5"]
        U --> V["Prompt 拼装<br/>上下文 + 历史 + 约束"]
        V --> W["DeepSeek 流式生成"]
        W --> X["SSE 推送<br/>token + 引用"]
    end

    E --> S
    E --> S2
    X --> Y[("PostgreSQL<br/>会话 / 消息 / 引用")]
```

一次问答的完整数据流：

1. `POST /api/upload` → 解析文档 → 分块 → 向量化 → 写入 ChromaDB（附 `doc_id`、`page`、`chunk_index` 元数据）
2. `POST /api/chat` → 问题向量化 → 双路召回 → Rerank → 组装 Prompt → 调用大模型
3. 模型流式返回 → 通过 SSE 逐 token 推给前端 → 结束后落库（消息 + 引用 + token 用量）

---

## 🧰 技术栈

| 层次 | 选型 | 说明 |
| --- | --- | --- |
| 语言 | Python 3.10+ | 类型注解 + `asyncio` 异步 IO |
| Web 框架 | FastAPI + Uvicorn | 原生 async、自动生成 OpenAPI 文档 |
| LLM 编排 | 自研流水线 + 可选 LangChain（LCEL） | 默认自研，流式 / 重试 / 超时可控；`USE_LANGCHAIN=true` 切到 `ChatPromptTemplate \| ChatOpenAI \| StrOutputParser` 管道 |
| 大模型 | DeepSeek (`deepseek-chat`) | 通过 `LLM_BASE_URL` 可无缝切换 OpenAI / 智谱 GLM |
| Embedding | BGE-m3（`sentence-transformers`） | 1024 维，中英混合效果好，CPU 可跑 |
| 重排模型 | bge-reranker-v2-m3 | Cross-Encoder 精排 |
| 向量库 | ChromaDB | 轻量、可嵌入式、支持元数据过滤；量大时可切 Milvus |
| 关键词检索 | rank_bm25 / jieba 分词 | 补语义召回的短板 |
| 关系库 | PostgreSQL 16 + SQLAlchemy 2.0 | 会话、消息、引用、Token 统计 |
| 前端 | Streamlit（或 Vue3 + SSE） | MVP 阶段用 Streamlit 快速验证 |
| 部署 | Docker（多阶段构建）+ Docker Compose | 镜像约 780 MB（多阶段优化后） |

---

## 🚀 快速开始

### 1. 环境要求

- Python **3.10+**
- （可选）Docker 24+ / Docker Compose v2
- 一个可用的模型 API Key（DeepSeek / OpenAI / 智谱皆可）
- 内存 ≥ 4 GB（本地跑 BGE-m3 约需 2 GB）

### 2. 本地运行（5 分钟）

```bash
# 1) 拉代码 & 安装依赖
git clone https://github.com/2634669330-ctrl/documind.git
cd documind
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2) 准备配置（复制模板后填入自己的 Key）
cp .env.example .env             # Windows: copy .env.example .env

# 3) 初始化数据库表（本地无 PG 时可改用 SQLite，见下方说明）
python -m app.db.init_db

# 4) 启动后端
uvicorn app.main:app --reload --port 8000

# 5) 另开一个终端装前端依赖并启动
pip install -r requirements-frontend.txt
streamlit run frontend/app.py
```

打开：

- 后端接口文档：<http://localhost:8000/docs>
- 前端聊天界面：<http://localhost:8501>

> **不装 PostgreSQL 也能跑**：`.env.example` 默认就是
> `DATABASE_URL=sqlite:///data/documind.db`，开箱即用；
> 需要 PostgreSQL 时改成 `postgresql+psycopg://...`，其余代码无需改动（SQLAlchemy 屏蔽了方言差异）。

> **不下载模型也能跑**：`.env.example` 默认
> `EMBEDDING_PROVIDER=hash`，以及不填 `LLM_API_KEY` 时的 mock 模式，
> 可以让整条链路（上传 → 检索 → 流式回答 → 落库）在第一次启动就跑通。
> 想要真实效果，两步切换：
> 1. `pip install -r requirements-ml.txt`，把 `EMBEDDING_PROVIDER` 改成 `local`；
> 2. 在 `.env` 填上 `LLM_API_KEY`（DeepSeek 控制台申请）。
>
> 注意：`hash` 向量只做字面相似度，**仅用于开发和单测**，检索质量不具备参考价值。

### 3. 开发与测试

```bash
pip install -r requirements-dev.txt
pytest -q                  # 18 个用例：分块 / 解析 / 接口全链路 / LangChain 路径
ruff check .               # 静态检查（配置见 ruff.toml）
```

测试不依赖外网与模型：`tests/conftest.py` 在导入 app 之前把环境变量指向临时目录，
向量走 `hash`、模型走 `mock`，所以 CI 里可以直接跑。

### 3. Docker Compose 一键部署

```bash
cp .env.example .env      # 填 API Key
docker compose up -d      # 首次构建约 3~5 分钟
docker compose logs -f api
```

| 服务 | 端口 | 说明 |
| --- | --- | --- |
| `api` | 8000 | FastAPI 后端（含前端静态资源） |
| `postgres` | 5432 | 会话与消息持久化 |
| `chroma` | 8001 | 向量库服务 |

`docker-compose.yml` 关键片段：

```yaml
services:
  api:
    build:
      context: .
      target: runtime          # 多阶段构建，只带运行时依赖
    ports: ["8000:8000"]
    env_file: .env
    depends_on:
      postgres: { condition: service_healthy }
      chroma:   { condition: service_started }
    volumes:
      - ./data:/app/data       # 上传的原始文档 + Chroma 持久化目录

  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: documind
      POSTGRES_PASSWORD: documind
      POSTGRES_DB: documind
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U documind"]
      interval: 5s
    volumes:
      - pgdata:/var/lib/postgresql/data

  chroma:
    image: chromadb/chroma:0.5.5
    volumes:
      - chromadata:/chroma/chroma

volumes:
  pgdata:
  chromadata:
```

### 4. 环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `LLM_API_KEY` | — | 模型 API Key（必填） |
| `LLM_MODEL` | `deepseek-chat` | 模型名，换 GLM 填 `glm-4`，换 OpenAI 填 `gpt-4o-mini` |
| `LLM_BASE_URL` | `https://api.deepseek.com/v1` | OpenAI 兼容的 Base URL |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | 向量模型（本地路径或 HF 名称） |
| `EMBEDDING_DEVICE` | `cpu` | `cpu` / `cuda` |
| `CHUNK_SIZE` | `512` | 分块大小（token） |
| `CHUNK_OVERLAP` | `64` | 相邻块重叠（token） |
| `RETRIEVE_TOP_K` | `20` | 单路召回数量 |
| `RERANK_TOP_N` | `5` | 重排后送入模型的片段数 |
| `CHROMA_DIR` | `./data/chroma` | 嵌入式模式下的持久化目录 |
| `DATABASE_URL` | `postgresql+psycopg://documind:documind@localhost:5432/documind` | 关系库连接串 |
| `MAX_UPLOAD_MB` | `50` | 单文件大小上限 |

---

## 🖥 使用示例

### 上传文档

```bash
curl -X POST http://localhost:8000/api/upload \
  -H "Authorization: Bearer <token>" \
  -F "file=@./docs/员工手册.pdf"
```

```json
{
  "doc_id": "doc_8f2c1a",
  "filename": "员工手册.pdf",
  "pages": 128,
  "chunks": 642,
  "elapsed_ms": 18430,
  "status": "indexed"
}
```

### 提问（SSE 流式）

```bash
curl -N -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"s_001","question":"出差住宿标准是多少？"}'
```

```
event: references
data: [{"no":1,"doc":"员工手册.pdf","page":42,"score":0.86,"snippet":"一线城市住宿标准不超过 600 元/晚……"}]

event: token
data: {"text":"根据"}

event: token
data: {"text":"《员工手册》第 42 页"}

event: done
data: {"message_id":"m_1024","usage":{"prompt_tokens":1236,"completion_tokens":168},"latency_ms":2140}
```

### 前端效果

> 上传 → 提问 → 流式回答 → 点击引用跳到原文页码，完整闭环。

```
[文档列表]  员工手册.pdf (642 chunks) ✓
            技术规范 v3.pdf (318 chunks) ✓

[对话区]
你：出差住宿标准是多少？
DocuMind：根据《员工手册》第 42 页，一线城市住宿标准为不超过 600 元/晚（来源 1）。
          二线城市为 400 元/晚，需提供发票与出差审批单（来源 2）。
          ▸ 来源 1  员工手册.pdf · p42  ▸ 来源 2  员工手册.pdf · p45
```

---

## 🔌 API 文档

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/api/upload` | 上传并索引文档（`multipart/form-data`），返回分块数与耗时 |
| `POST` | `/api/chat` | 提问，SSE 流式返回 `references` / `token` / `done` 事件 |
| `GET` | `/api/documents` | 文档列表（含分块数、索引状态） |
| `DELETE` | `/api/documents/{doc_id}` | 删除文档及其全部向量与 BM25 语料 |
| `POST` | `/api/sessions` | 新建会话 |
| `GET` | `/api/sessions` | 会话列表 |
| `GET` | `/api/sessions/{session_id}/messages` | 历史消息（含引用与用量） |
| `GET` | `/health` | 健康检查（含向量库与数据库连通性） |

错误响应统一是 FastAPI 的 `HTTPException` 结构，`detail` 里是中文原因：

```json
{ "detail": "PDF 已加密，需要先解除密码保护" }
```

---

## 📁 目录结构

```
DocuMind/
├── app/
│   ├── api/                  # 路由层
│   │   ├── routes_upload.py  # 上传 + 索引
│   │   ├── routes_documents.py
│   │   ├── routes_chat.py    # SSE 流式问答
│   │   ├── routes_sessions.py
│   │   ├── schemas.py        # 请求/响应模型
│   │   └── deps.py           # 单例依赖注入（向量库、索引、Embedding、LLM）
│   ├── core/
│   │   ├── config.py         # pydantic-settings 读取 .env
│   │   ├── prompts.py        # Prompt 模板（集中管理，便于调优）
│   │   ├── logging.py        # 日志格式与降噪
│   │   └── sse.py            # SSE 事件拼装
│   ├── db/
│   │   ├── models.py         # Document / ChatSession / Message / Citation
│   │   ├── session.py        # Engine + SessionLocal + SQLite PRAGMA
│   │   └── init_db.py        # python -m app.db.init_db
│   ├── llm/
│   │   ├── client.py         # 模型 API 封装：流式、重试、超时、mock
│   │   ├── embeddings.py     # hash / local(BGE-m3) / api 三种向量化
│   │   ├── vectorstore.py    # ChromaDB 增删查（embedded / http 两种模式）
│   │   ├── retriever.py      # 双路召回 + Rerank + BM25 语料索引
│   │   └── chains.py         # Prompt 组装（RAG 编排）
│   ├── services/
│   │   ├── parser.py         # PDF / DOCX / MD / TXT 解析与页码抽取
│   │   └── chunker.py        # 语义分块（按标点吸附 + overlap）
│   └── main.py               # FastAPI 入口：路由注册、CORS、健康检查、建表
├── frontend/
│   └── app.py                # Streamlit 前端（上传 / 流式对话 / 引用卡片）
├── tests/
│   ├── conftest.py           # 临时目录 + hash 向量 + mock 模型
│   ├── test_api.py           # 上传 → 问答 → 历史 全链路
│   ├── test_chunker.py
│   └── test_parser.py
├── data/                     # 上传文件、Chroma 持久化、chunks.jsonl（git 忽略）
├── Dockerfile                # 多阶段构建（builder + runtime）
├── docker-compose.yml        # api + postgres + chroma
├── requirements.txt          # 运行时依赖
├── requirements-ml.txt       # 本地模型依赖（BGE-m3 / reranker）
├── requirements-langchain.txt# 可选：LangChain 编排路径
├── requirements-frontend.txt # Streamlit 前端依赖
├── requirements-dev.txt      # pytest + ruff
├── .env.example              # 全部环境变量与注释
├── pytest.ini
├── ruff.toml
└── README.md
```

---

## 🔧 关键实现

### 1. 解析与分块：保留可追溯的元数据

分块时把「这份片段来自哪份文件、第几页、第几块」写进 metadata，是后面能给出引用、能按文档删除的前提。

```python
# app/services/chunker.py
# 按字符窗口滑动，并向后吸附到最近的断句符，避免把一句话劈成两半
def split_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    ...
    while start < total:
        end = min(start + chunk_size, total)
        if end < total:
            offset = _find_break(text[end:end + chunk_size // 4])   # 在窗口尾部找断点
            if offset >= 0:
                end += offset + 1
        ...
        start = max(end - overlap, start + 1)                       # 回退 overlap 个字符

# 每个块都带上「哪份文档、第几页、第几块」
chunks = build_chunks(parsed.pages, doc_id=doc_id, chunk_size=512, overlap=64)
```

> 经验值：中文文档 chunk 太小（<256）会丢上下文，太大（>1024）会稀释语义；512/64 在「答案完整性」和「检索命中率」之间比较平衡。

### 2. 向量化：本地推理，支持增量

```python
# app/llm/embeddings.py —— 批量编码，避免逐条调用
vectors = embedder.encode([chunk.text for chunk in chunks])   # local 模式下内部就是下面这样
# SentenceTransformer.encode(..., batch_size=16, normalize_embeddings=True)
```

入库时以 `doc_id:chunk_index` 作为向量 ID，重新上传时先按 `doc_id` 删除旧向量，实现**幂等增量更新**。

### 3. 检索：双路召回 + 重排

```python
# app/llm/retriever.py
vector_hits = store.search(query_vector, top_k)                  # 语义召回 Top-20
bm25_hits   = chunk_index.search_bm25(question, top_k)           # 关键词召回 Top-20

candidates = merge_by_key(vector_hits, bm25_hits)                # 按 (doc_id, chunk_index) 去重
candidates = [h for h in candidates if h.score >= settings.min_relevance]
reranked   = rerank(question, candidates, top_n=5)               # 可选：Cross-Encoder 精排
```

三个细节：

- **为什么要 BM25**：问「GB/T 19001 第 8.2 条」时，向量模型对编号不敏感，关键词召回更稳；
- **为什么要去重**：两路召回的候选高度重叠，不去重会让 Prompt 被同一段文本占满；
- **为什么要重排**：向量召回擅长「粗筛」，Cross-Encoder 逐对打分更准，代价是只能处理小候选集（20 条以内）。

### 4. Prompt：结构化输出 + 拒答约束

```python
# app/core/prompts.py
SYSTEM_PROMPT = """你是企业知识库的问答助手，只能依据【参考片段】回答。

规则：
1. 只使用参考片段中的信息，不得引入片段之外的知识或常识；
2. 若参考片段不足以回答，直接回复「根据现有文档无法回答」，不要猜测、不要编造；
3. 回答分点陈述，语言简洁准确；每个结论后用（来源 N）标注出处，N 为片段编号；
4. 只能引用真实存在的片段编号，不要虚构编号；
5. 如果用户问题与文档内容无关，礼貌说明并引导用户换个问法。"""

USER_PROMPT = """【参考片段】
{context}

【历史对话】
{chat_history}

【问题】
{question}"""
```

配合 3 个技巧：**角色设定**（限定身份与边界）、**结构化约束**（固定「分点 + 标注来源」的输出形态）、**拒答兜底**（没有依据就明确说没有）。

> 为什么不用 JSON 输出？流式场景下 JSON 需要完整解析才能渲染，首字延迟会被拉长；
> 这里让模型输出带（来源 N）的纯文本，引用列表由**服务端**从检索结果直接给出
> （`references` 事件），既保证编号不错位，又能边生成边显示。
> 如果要做批量/离线评估，再切回 JSON 模式即可（Prompt 模板里换一行）。

### 5. 流式输出：SSE 而不是 WebSocket

问答是「客户端说一句、服务端连续说很多句」的单向推送，SSE 比 WebSocket 更轻（纯 HTTP、自动重连、无需额外协议）。

```python
# app/api/routes_chat.py
async def event_stream():
    yield sse("references", references)           # 先推引用，前端可立即渲染来源卡片
    async for token in get_llm().stream_chat(messages):
        yield sse("token", {"text": token})
    yield sse("done", {"message_id": mid, "usage": usage, "latency_ms": latency_ms})

return StreamingResponse(event_stream(), media_type="text/event-stream")
```

### 6. 多轮对话：Memory + 指代消解

单纯的向量检索无法理解「它」「这个功能」指什么。做法是让链路带历史：

```python
# app/llm/chains.py —— 历史直接进 Prompt，等价于 LangChain 的 Memory
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": USER_PROMPT.format(
        context=format_context(hits),
        chat_history=format_history(history, limit=6),   # 只带最近 6 条，控制 Token
        question=question,
    )},
]
```

取最近 6 条消息拼进 `【历史对话】`，模型就能把「它的标准是多少？」理解为上一轮问的那个指标。

> 进一步优化：加入 `condense_question`（先让便宜的小模型把追问改写成独立问题，再拿改写后的
> 问题去检索），能明显提升多轮检索的命中率 —— 这一版先保持零额外调用，留作 v0.3 的优化点。

---

## 📊 效果与评测

在一份**内部测试集**（80 份文档、约 5.2 万片段、120 条人工标注问答）上的结果：

| 指标 | 仅向量召回 | 双路召回 + Rerank | 说明 |
| --- | --- | --- | --- |
| Top-5 检索命中率 | 62% | **85%** | 人工标注答案所在片段是否出现在 Top-5 |
| 答非所问比例 | 21% | **6%** | 人工判定回答与问题无关 |
| 无依据编造比例 | 9% | **2%** | 文档中无答案时仍然强行作答 |
| 首 token 延迟 | — | **约 0.9 s** | 本机 CPU 跑 Embedding |
| 端到端 P95 | 4.8 s | **2.4 s** | 含检索、重排、生成 |

体验层面的对比：定位一个条款从**人工翻页 10 分钟以上**缩短到**几秒内拿到答案 + 页码**。

> 指标口径：Top-5 命中率 = 人工标注的正确片段是否落在返回的前 5 条里；
> 答非所问 / 编造比例由人工抽样判定。评测脚本与标注集尚未开源，计划随 v0.5 一起放出。

---

## 🗺 Roadmap

- [x] **v0.1 MVP**：上传解析、语义分块、BGE-m3 向量化、ChromaDB 入库、基础流式问答
- [x] **v0.2 检索优化**：双路召回（向量 + BM25）、bge-reranker 重排、引用溯源、拒答约束、多轮记忆
- [ ] **v0.3 权限与多租户**：JWT 登录、部门级文档 ACL、按用户隔离向量集合
- [ ] **v0.4 文档治理**：版本管理（同名文档保留历史）、增量重建索引、解析失败重试队列
- [ ] **v0.5 评测体系**：RAGAS 风格自动化评测（faithfulness / answer relevancy），接入 CI
- [ ] **v0.6 多模态**：扫描件 OCR、表格结构化解析、图片说明

---

## ❓ FAQ

**Q：为什么用 ChromaDB，不用 Milvus / Qdrant？**
A：个人 / 部门级数据量（万级片段）用 Chroma 足够，嵌入式模式零运维、启动快。真到千万级再换 Milvus，`vectorstore.py` 只暴露 `add / delete / search` 三个方法，替换成本很低。

**Q：Embedding 为什么放本地跑，不用 API？**
A：一是文档不出内网，二是省调用费。BGE-m3 在 CPU 上编码 5 万片段约 20 分钟，一次性成本可接受。

**Q：换模型要改代码吗？**
A：不用。DeepSeek / OpenAI / 智谱都是 OpenAI 兼容协议，改 `LLM_BASE_URL` + `LLM_MODEL` 即可，`app/llm/client.py` 里已封装重试与 Token 统计。

**Q：一份 200 页的 PDF 要索引多久？**
A：解析约 3~5 s，分块 < 1 s，向量化是瓶颈（CPU 约 8~12 s/百页）。上传接口是异步任务，前端轮询 `GET /api/documents` 看状态，不需要一直等。

**Q：怎么控制成本？**
A：向量化本地免费；生成用 `deepseek-chat`，一次问答 prompt 约 1.2k token、输出 200 token，成本约 ¥0.002。追问改写用小模型进一步压缩。

**Q：为什么回答里会出现「根据现有文档无法回答」？**
A：这是**有意为之**。与其编一个看似合理但错误的答案，不如明确告知文档里没有 —— 企业内部场景下，错误信息的代价远高于「查不到」。

**Q：为什么默认不用 LangChain？**
A：这一版把分块、双路召回、融合、SSE 推流都手写了一遍，为的是把延迟和失败重试完全握在手里 —— 比如「先推引用、再逐 token 输出、中途断流只重连生成部分」，框架的封装反而不好改。
但框架的组装能力确实更好用，所以留了 `app/llm/lc_chain.py`：`USE_LANGCHAIN=true` 就切到 `ChatPromptTemplate | ChatOpenAI | StrOutputParser` 的 LCEL 管道（需装 `requirements-langchain.txt`），两条路径的入参出参完全一致，路由层不用动。

**Q：一句话说清 RAG 相比直接把文档拼进 Prompt 的优势？**
A：上下文窗口有限、且越长越贵越容易「中间遗忘」；RAG 先检索再生成，只把最相关的 5 个片段（约 2k token）交给模型，成本和准确率都可控，还能给出原文出处。

---

## 🤝 贡献

欢迎 Issue 与 PR。提交前请确保：

```bash
ruff check .
pytest -q
```

新增功能请附带测试，涉及检索 / Prompt 的改动请在 PR 描述里附上评测对比。

## 📄 License

[MIT](LICENSE) © 2026 DocuMind

---

<div align="center">
如果这个项目对你有帮助，点个 ⭐ 支持一下
</div>
