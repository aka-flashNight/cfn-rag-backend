# 离线插话假流服务（仅测试 / 联调）

给游戏窗口做真实 HTTP + SSE 联调用，**不接真模型、不消耗 API 配额、不进入生产默认路径**。
实现见 `fake_interject_server.py`；Uvicorn 真实监听，端点与生产一致。

## 启动

在仓库根目录：

```powershell
.\venv\Scripts\Activate.ps1
python -m tests.integration.fake_interject_server --port 7088 --fixture auto
```

默认端口 `7088`；`--data-dir` 可指定 session 库目录，缺省自动建系统临时目录（不写 `tools/memory.db`）。

## 端点

```text
POST http://127.0.0.1:7088/api/game/dialogue-interject
POST http://127.0.0.1:7088/api/game/dialogue-interject/finalize
GET  http://127.0.0.1:7088/api/health
POST http://127.0.0.1:7088/api/integration/bind      # 仅内存，不建槽目录/不写存档
POST http://127.0.0.1:7088/api/integration/unbind
GET  http://127.0.0.1:7088/api/integration/status
```

`bind` 只写进程内 `IntegrationState`，不调用 `ensure_agent_save_files`、不刷新索引。

## fixture 选择

| 方式 | 行为 |
|---|---|
| `--fixture auto`（默认） | task → `F-09`；`player_input` 含「零有效」→ `F-10`；含「部分」→ `F-11`；含「北边」→ `F-02`；其余 → `F-01` |
| `--fixture F-01` / `F-02` / `F-09` / `F-10` / `F-11` | 所有请求强制回放该 fixture（F-09 需要发 task 请求才看得到 mute） |
| `--chunk-delay-ms N` | 每个 chunk 之间延迟 N 毫秒，便于观察逐句到达与断连取消 |

## 断连与 finalize

- 客户端直接关闭 SSE 连接即可验证真取消：服务端会取消在途假流，不写 AI 台词。
- 关闭后照常调用 `finalize`，`transcript` 传截至当时的全部已播行；重复调用幂等。
- 本服务只做内存 bind，不读写游戏存档；插话 session 的 `slot_key` 取 bind 的槽位键。

## 测试

```powershell
python -m pytest tests/integration/ -q
```

`test_http_fake_server.py` 覆盖 F-01/09/10/11、finalize 幂等与真实 HTTP 断连取消。
