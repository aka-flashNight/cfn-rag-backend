"""api 测试共享夹具。

B2-4 起 bind 端点会调用 `core.startup.ensure_agent_save_files`，其落点是
「资源目录」（开发环境解析到真实游戏仓根）。API 测试一律不得写真实仓：
autouse 夹具把 `api.integration_api` 里导入的同名引用默认替换为 no-op；
需要观测接线的用例可在自己测试体内再次 monkeypatch 覆盖（后设的先生效）。
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_agent_save_ensure(monkeypatch):
    from api import integration_api

    monkeypatch.setattr(
        integration_api, "ensure_agent_save_files", lambda slot_key: None
    )
