# CFN-RAG Backend 部署指南

## 双模式架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          CFN-RAG Backend                                 │
│                                                                          │
│  ┌──────────────────────────┐    ┌──────────────────────────────────┐   │
│  │   Local Profile           │    │   Server Profile                 │   │
│  │   (双击 exe / python main)│    │   (Docker / K8s)                 │   │
│  │                           │    │                                  │   │
│  │  • SQLite (memory.db)     │    │  • PostgreSQL (RDS)              │   │
│  │  • LlamaIndex 本地索引     │    │  • Qdrant (向量数据库)           │   │
│  │  • In-Memory Cache        │    │  • Redis (缓存 / 任务队列)       │   │
│  │  • 无外部依赖              │    │  • arq Worker (后台任务)         │   │
│  │  • 一键启动               │    │  • Prometheus /metrics           │   │
│  └──────────────────────────┘    │  • JSON 结构化日志                │   │
│                                   └──────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

## 环境准备

### 1. 前置依赖

| 工具 | 用途 | 安装 |
|------|------|------|
| Docker Desktop | 容器运行时 | [docker.com](https://www.docker.com/products/docker-desktop/) |
| WSL2 | Linux 内核（Windows 必需） | `wsl --install` |
| kubectl | K8s 命令行 | [kubernetes.io](https://kubernetes.io/docs/tasks/tools/) |
| Helm | K8s 包管理 | [helm.sh](https://helm.sh/docs/intro/install/) |
| minikube | 本地 K8s 集群 | [minikube.sigs.k8s.io](https://minikube.sigs.k8s.io/docs/start/) |

### 2. 验证环境

```powershell
# Windows PowerShell
docker --version
wsl --version
kubectl version --client
helm version
minikube version
```

---

## 方式一：Docker Compose（开发/测试推荐）

### 仅启动依赖服务（Redis + Qdrant + Postgres）

后端仍在本地 `python main.py` 运行，方便调试：

```bash
docker-compose up -d redis qdrant postgres
```

### 全栈启动

```bash
docker-compose --profile full up -d
```

### 访问服务

| 服务 | 地址 |
|------|------|
| API | http://localhost:7077/api/health |
| Prometheus | http://localhost:7077/metrics |
| Qdrant Dashboard | http://localhost:6333/dashboard |
| Jaeger UI | http://localhost:16686 |

---

## 方式二：minikube（K8s 本地集群）

### 1. 启动 minikube

```powershell
minikube start --cpus=4 --memory=8192 --driver=docker
```

### 2. 挂载本地模型和资源目录

```powershell
# 需要另开一个终端保持运行
minikube mount ./models:/mnt/models &
minikube mount ./resources:/mnt/resources &
```

### 3. 创建 PV/PVC

```bash
kubectl apply -f deploy/k8s/base/pv-models.yaml
kubectl apply -f deploy/k8s/base/pv-resources.yaml
```

### 4. 部署应用

```bash
# 应用所有 K8s 清单
kubectl apply -f deploy/k8s/base/

# 等待就绪
kubectl wait --for=condition=ready pod -l app=cfn-rag-backend --timeout=120s
```

### 5. 访问服务

```bash
# 端口转发到本地
kubectl port-forward svc/cfn-rag-backend 7077:7077
```

浏览器打开: http://localhost:7077/api/health

### 6. 清理

```bash
kubectl delete -f deploy/k8s/base/
minikube stop
```

---

## 方式三：Helm Chart

```bash
# 安装
helm install cfn-rag deploy/helm/cfn-rag -f deploy/helm/cfn-rag/values.local.yaml

# 升级
helm upgrade cfn-rag deploy/helm/cfn-rag -f deploy/helm/cfn-rag/values.local.yaml

# 卸载
helm uninstall cfn-rag
```

---

## 配置文件说明

### 关键环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `CFN_PROFILE` | `local` | 部署模式：`local` 或 `server` |
| `CFN_CACHE_BACKEND` | `memory` | 缓存后端：`memory` 或 `redis` |
| `CFN_DB_BACKEND` | `sqlite` | 数据库：`sqlite` 或 `postgres` |
| `CFN_VECTOR_BACKEND` | `llamaindex_local` | 向量库：`llamaindex_local` 或 `qdrant` |
| `CFN_CHECKPOINT_BACKEND` | `sqlite` | Checkpoint：`sqlite` / `postgres` / `redis` |
| `CFN_REDIS_URL` | `redis://localhost:6379/0` | Redis 连接地址 |
| `CFN_POSTGRES_URL` | `postgresql+asyncpg://...` | Postgres 连接地址 |
| `CFN_QDRANT_URL` | `http://localhost:6333` | Qdrant 连接地址 |
| `CFN_WORKER_BROKER_URL` | `redis://localhost:6379/1` | Worker 消息队列 |

### Profile 自动推导

当 `CFN_PROFILE=server` 时，以下后端自动切换（除非显式设置覆盖）：

- `cache_backend`: memory → **redis**
- `db_backend`: sqlite → **postgres**
- `vector_backend`: llamaindex_local → **qdrant**
- `checkpoint_backend`: sqlite → **postgres**

---

## 故障排查

### Docker Desktop 未运行

```
error during connect: Get "http://%2F%2F.%2Fpipe%2Fdocker_engine/v1.24/containers/json":
open //./pipe/docker_engine: The system cannot find the file specified.
```

解决：启动 Docker Desktop，等待鲸鱼图标静止。

### minikube 启动失败

```powershell
# 重置 minikube
minikube delete
minikube start --driver=docker --cpus=4 --memory=8192
```

### 端口冲突

```powershell
# 检查 7077 端口占用
netstat -ano | findstr :7077
```

### Qdrant connection refused (server profile)

确保 Qdrant 服务已启动：
```bash
docker-compose up -d qdrant
# 或
kubectl get pods -l app=qdrant
```

### Postgres 连接超时

检查 Postgres 健康状态：
```bash
docker-compose ps postgres
# 查看日志
docker-compose logs postgres
```
