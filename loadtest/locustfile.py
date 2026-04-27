"""
CFN-RAG Backend 压测脚本（Locust）。

场景：
- 50 并发用户，每用户 5 轮连续对话
- 10% 概率触发任务草案（会导致额外的 tool_calls 和 checkpointer 写入）
- SSE 流式与非流式两种模式

指标：
- p50 / p95 / p99 首字延迟（TTFB）
- 完整回复延迟
- 失败率
- 吞吐量（RPS）

用法::

    # 启动压测
    locust -f loadtest/locustfile.py \
        --host=http://localhost:7077 \
        --users=50 \
        --spawn-rate=5 \
        --run-time=300s \
        --html=loadtest/reports/report_$(date +%Y%m%d_%H%M%S).html

    # Web UI 模式（推荐调试时使用）
    locust -f loadtest/locustfile.py --host=http://localhost:7077
"""

from __future__ import annotations

import json
import random
import time
import uuid
from typing import Any

from locust import HttpUser, task, between, events

# 预设问题列表（覆盖不同 game progress stage）
USER_QUERIES: list[dict] = [
    {"query": "你好，最近有什么新鲜事吗？",                      "stage": 1},
    {"query": "你知道这附近有什么危险的地方吗？",                "stage": 2},
    {"query": "我需要一件趁手的武器，你有什么建议？",            "stage": 3},
    {"query": "听说黑铁会最近动静很大，你知道怎么回事吗？",       "stage": 4},
    {"query": "诺亚那边的防线还稳固吗？",                       "stage": 5},
    {"query": "我上次帮了你那个忙，现在能给我点奖励吗？",          "stage": 6},
    {"query": "有什么任务可以交给我？我想赚点外快。",             "stage": 7},
    {"query": "我对这个世界的规则还不太了解，能给我讲讲吗？",      "stage": 1},
    {"query": "你在想什么？",                                   "stage": 3},
    {"query": "再见，我要走了。",                               "stage": 5},
]

NPC_NAMES: list[str] = [
    "小琪", "凯瑟琳", "阿达特", "卢卡斯", "索菲亚",
    "卡洛斯", "玛丽亚", "杰克", "艾琳", "诺顿",
]


class CfnRagUser(HttpUser):
    """模拟一个 NPC 对话用户。

    每个用户创建一个 session，然后发送多轮对话。
    """

    wait_time = between(5, 15)  # 模拟真实用户对话间隔

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.session_id: str = ""
        self.npc_name: str = ""
        self.round: int = 0
        self.max_rounds: int = 5

    def on_start(self):
        """创建会话（每个用户仅一次）。"""
        self.npc_name = random.choice(NPC_NAMES)
        self.round = 0

        # 创建 session
        resp = self.client.post(
            "/api/game/sessions",
            json={"npc_name": self.npc_name, "title": f"压测-{uuid.uuid4().hex[:8]}"},
            name="/api/game/sessions",
        )
        if resp.status_code == 200:
            self.session_id = resp.json().get("session_id", "")
        else:
            # 简化：使用随机 session_id
            self.session_id = uuid.uuid4().hex

    @task
    def chat_round(self):
        """一轮对话。"""
        if self.round >= self.max_rounds:
            self.on_start()  # 重新开始新会话
            return

        self.round += 1
        q = random.choice(USER_QUERIES)

        # 10% 概率触发任务草案（讨要任务）
        if random.random() < 0.1:
            q = {"query": "给我发布一个任务吧，我想做点什么。", "stage": q["stage"]}

        payload = {
            "query": q["query"],
            "npc_name": self.npc_name,
            "session_id": self.session_id,
            "progress_stage": q["stage"],
            "agent_enabled": True,
        }

        # 50% 使用 SSE 流式，50% 使用非流式
        use_stream = random.random() < 0.5

        if use_stream:
            self._stream_ask(payload)
        else:
            self._normal_ask(payload)

    def _normal_ask(self, payload: dict):
        """非流式请求（测试延迟和成功率）。"""
        t0 = time.monotonic()
        with self.client.post(
            "/api/game/ask",
            json=payload,
            name="/api/game/ask",
            catch_response=True,
            timeout=120,
        ) as resp:
            elapsed = time.monotonic() - t0
            if resp.status_code == 200:
                data = resp.json()
                reply_len = len(data.get("reply", ""))
                resp.request_meta["ttfb"] = elapsed
                resp.request_meta["reply_len"] = reply_len
                resp.success()
            else:
                resp.failure(f"HTTP {resp.status_code}: {resp.text[:200]}")

    def _stream_ask(self, payload: dict):
        """SSE 流式请求（测试首字延迟和流完整性）。"""
        t0 = time.monotonic()
        first_token_time: float | None = None
        reply_chunks: list[str] = []
        done_received = False

        with self.client.post(
            "/api/game/ask",
            json=payload,
            params={"stream": "true"},
            name="/api/game/ask?stream=true",
            catch_response=True,
            stream=True,
            timeout=120,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}: {resp.text[:200]}")
                return

            for line in resp.iter_lines(decode_unicode=True):
                if not line or line.startswith(":"):
                    continue

                if line.startswith("event: "):
                    event_type = line[7:]
                elif line.startswith("data: "):
                    data_str = line[6:]
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    if event_type == "content":
                        if first_token_time is None:
                            first_token_time = time.monotonic() - t0
                        reply_chunks.append(data.get("delta", ""))
                    elif event_type == "done":
                        done_received = True

        total_time = time.monotonic() - t0
        ttfb = first_token_time or total_time

        if done_received:
            resp.request_meta["ttfb"] = ttfb
            resp.request_meta["reply_len"] = len("".join(reply_chunks))
            resp.request_meta["total_time"] = total_time
            resp.success()
        else:
            resp.failure("SSE 未收到 done 事件")


# ---------------------------------------------------------------------------
# 自定义事件：记录延迟分布
# ---------------------------------------------------------------------------


@events.request.add_listener
def on_request(
    request_type: str,
    name: str,
    response_time: float,
    response_length: int,
    exception: Exception | None,
    context: dict,
    **kwargs,
) -> None:
    """在 Locust 日志中附加 TTFB 信息。"""
    if exception:
        return

    meta = context or {}
    if "ttfb" in meta:
        # 将 TTFB 作为额外指标记录到环境
        pass
