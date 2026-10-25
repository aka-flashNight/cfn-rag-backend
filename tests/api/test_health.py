"""health 接口测试（B1-2）：集成字段 + 原字段兼容。"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.integration.state import get_integration_state


def _make_client() -> TestClient:
    from api import api_router

    app = FastAPI()
    app.include_router(api_router, prefix="/api")
    return TestClient(app)


@pytest.fixture
def state():
    st = get_integration_state()
    original_launch = st.launch_mode
    st.unbind()
    st.launch_mode = "standalone"
    yield st
    st.unbind()
    st.launch_mode = original_launch


def test_health_fields_and_compat(state):
    body = _make_client().get("/api/health").json()
    assert body["status"] == "ok"
    assert body["app"] == "CFN-RAG Backend"  # 原字段保留兼容
    assert body["version"] == "3.0.0"
    assert body["mode"] == "standalone"
    assert body["bound_slot"] is None
    assert body["frontend_url"] == "http://127.0.0.1:7077/"


def test_health_reflects_bind(state):
    client = _make_client()
    client.post("/api/integration/bind", json={"slot_key": "slot_health"})
    body = client.get("/api/health").json()
    assert body["mode"] == "embedded"
    assert body["bound_slot"] == "slot_health"
    assert body["frontend_url"] == "http://127.0.0.1:7077/?embedded=1"
