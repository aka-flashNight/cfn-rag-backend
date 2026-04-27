# CFN-RAG Backend 压测

## 快速开始

```bash
# 安装 locust
pip install locust>=2.31

# Web UI 模式（推荐调试）
locust -f loadtest/locustfile.py --host=http://localhost:7077

# 命令行模式（CI / 自动化）
locust -f loadtest/locustfile.py \
    --host=http://localhost:7077 \
    --users=50 \
    --spawn-rate=5 \
    --run-time=300s \
    --html=loadtest/reports/report_$(date +%Y%m%d_%H%M%S).html \
    --csv=loadtest/reports/results
```

## 场景说明

| 场景 | 描述 |
|------|------|
| 常规对话 | 50 并发用户，5 轮随机闲聊（覆盖 stage 1-7） |
| 任务草案 | 10% 概率触发任务发布（tool_calls + checkpointer 写入） |
| SSE 流式 | 50% 概率使用 SSE 流式请求（测试首字延迟） |

## 目标指标

| 指标 | 目标（单 pod） | 目标（3 pods） |
|------|---------------|---------------|
| p50 首字延迟 | < 1.0s | < 0.8s |
| p95 首字延迟 | < 3.0s | < 1.5s |
| 失败率 | < 1% | < 1% |
| 并发数 | 20 | 50 |

## 报告解读

生成的 HTML 报告包含：
- RPS（每秒请求数）
- 响应时间分布（p50/p75/p95/p99）
- 失败率
- 各 endpoint 独立统计
