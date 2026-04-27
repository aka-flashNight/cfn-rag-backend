# ============================================================================
# CFN-RAG Backend 容器镜像（多阶段构建）
#
# 构建:
#   docker build -t cfn-rag-backend:latest .
#
# 运行:
#   docker run -p 7077:7077 \
#     -v $(pwd)/models:/app/models \
#     -v $(pwd)/resources:/app/resources \
#     --env-file .env \
#     cfn-rag-backend:latest
#
# 注意：
# - 模型文件（models/bge-small-zh-v1.5）通过卷挂载，不进镜像
# - 游戏数据（resources/）通过卷挂载，不进镜像
# - .env 中的 API key 通过 --env-file 或 K8s Secret 注入
# ============================================================================

# --- Builder stage: 安装依赖 ---
FROM python:3.11-slim AS builder

WORKDIR /app

# 安装构建工具
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 先复制 requirements 并安装（利用 Docker 层缓存）
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# --- Runtime stage: 最小运行镜像 ---
FROM python:3.11-slim AS runtime

WORKDIR /app

# 安装运行时系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 从 builder 复制已安装的包
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 复制源码
COPY core/ ./core/
COPY api/ ./api/
COPY services/ ./services/
COPY schemas/ ./schemas/
COPY ai_engine/ ./ai_engine/
COPY evals/ ./evals/
COPY worker/ ./worker/
COPY scripts/ ./scripts/
COPY alembic.ini .
COPY alembic/ ./alembic/
COPY main.py .

# 模型和游戏数据在运行时通过卷挂载
# 默认路径：/app/models /app/resources
RUN mkdir -p /app/models /app/resources

# 健康检查
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7077/api/health || exit 1

EXPOSE 7077

# 默认以 local profile 启动（单机模式）
# Server profile 通过环境变量 CFN_PROFILE=server 切换
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7077", "--log-level", "info"]
