"""LLM 连接配置 API：localStorage 持久化，后端只保留内存副本。"""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import api.game_api as game_api
import services.llm.recent as recent


@pytest.fixture(autouse=True)
def _isolate_memory(monkeypatch):
    monkeypatch.setattr(recent, "_recent", None)
    monkeypatch.setattr(recent, "_updated_at", 0)


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(game_api.router, prefix="/api/game")
    return TestClient(app)


def test_initial_state_is_unconfigured(client):
    body = client.get("/api/game/llm-config").json()
    assert body["ok"] is True
    assert body["configured"] is False
    assert body["api_key"] == ""
    assert body["updated_at"] == 0


def test_save_then_read_roundtrip(client):
    response = client.post("/api/game/llm-config", json={
        "api_key": "sk-test",
        "api_base": "https://api.deepseek.com/v1",
        "model_name": "deepseek-chat",
        "updated_at": 1234,
    })
    assert response.status_code == 200
    body = response.json()
    assert body["configured"] is True
    assert body["api_key"] == "sk-test"
    assert body["updated_at"] == 1234

    assert client.get("/api/game/llm-config").json()["model_name"] == "deepseek-chat"


def test_save_without_proxy_preserves_existing_proxy(client):
    client.post("/api/game/llm-config", json={
        "api_key": "sk-a", "api_base": "https://a.example/v1",
        "model_name": "m-a", "proxy_url": "http://127.0.0.1:7890",
    })
    body = client.post("/api/game/llm-config", json={
        "api_key": "sk-b", "api_base": "https://b.example/v1",
        "model_name": "m-b",
    }).json()
    assert body["api_key"] == "sk-b"
    assert body["proxy_url"] == "http://127.0.0.1:7890"


def test_clearing_key_replaces_memory_instead_of_falling_back(client):
    client.post("/api/game/llm-config", json={
        "api_key": "sk-old", "api_base": "https://a/v1", "model_name": "m",
    })
    body = client.post("/api/game/llm-config", json={
        "api_key": "", "api_base": "https://a/v1", "model_name": "m",
    }).json()
    assert body["configured"] is False
    assert "sk-old" not in json.dumps(body)


def test_no_disk_persistence(client, tmp_path):
    """显式断言不落盘：内存配置清空后重新查询必须回到未配置。"""
    client.post("/api/game/llm-config", json={
        "api_key": "sk-memory", "api_base": "https://a/v1", "model_name": "m",
    })
    assert recent.get_recent_llm_config() is not None
    recent._recent = None
    recent._updated_at = 0
    body = client.get("/api/game/llm-config").json()
    assert body["configured"] is False
