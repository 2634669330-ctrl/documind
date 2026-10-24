# syntax=docker/dockerfile:1
# 多阶段构建：builder 只负责把依赖编译成 wheel，runtime 只保留运行所需的包，
# 镜像体积能从 ~1.1GB 降到 ~700MB，且运行镜像里没有编译器和头文件。

FROM python:3.11-slim AS builder

WORKDIR /build
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY requirements.txt .
RUN pip wheel --wheel-dir /wheels -r requirements.txt


FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="DocuMind" \
      org.opencontainers.image.description="RAG based document knowledge base QA service" \
      org.opencontainers.image.source="https://github.com/2634669330-ctrl/documind"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# 只装本地 wheel，不走网络，构建更快也更可控
COPY --from=builder /wheels /wheels
COPY requirements.txt .
RUN pip install --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels

# 非 root 运行，降低容器逃逸风险
RUN useradd --create-home --uid 10001 appuser

COPY app ./app
COPY frontend ./frontend

RUN mkdir -p /app/data && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# 容器编排会用它判断服务是否真的可用（而不是只判断进程在不在）
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
