"""core.startup.ensure_agent_save_files 单测（B2-4）。

隔离手段：monkeypatch `core.startup.find_resources_directory` → tmp 资源根
（开发环境的真实值＝游戏仓根，直接跑会写真实仓，故所有用例必须走此夹具）。
"""

from __future__ import annotations

import json

import pytest

from core import startup
from services.integration.save_paths import agent_save_task_paths


@pytest.fixture
def resources_root(tmp_path, monkeypatch):
    monkeypatch.setattr(startup, "find_resources_directory", lambda: tmp_path)
    return tmp_path


def test_creates_slot_dir_and_empty_files(resources_root):
    startup.ensure_agent_save_files("cf7_abc")
    tasks_path, text_path = agent_save_task_paths(resources_root / "data", "cf7_abc")
    assert json.loads(tasks_path.read_text(encoding="utf-8")) == {"tasks": []}
    assert json.loads(text_path.read_text(encoding="utf-8")) == {}


def test_existing_content_untouched_missing_one_filled(resources_root):
    tasks_path, text_path = agent_save_task_paths(resources_root / "data", "cf7_abc")
    tasks_path.parent.mkdir(parents=True)
    tasks_path.write_text(
        json.dumps({"tasks": [{"id": 200001}]}, ensure_ascii=False), encoding="utf-8"
    )

    startup.ensure_agent_save_files("cf7_abc")

    assert json.loads(tasks_path.read_text(encoding="utf-8"))["tasks"][0]["id"] == 200001
    assert text_path.exists()


def test_invalid_slot_key_skips_without_write(resources_root):
    startup.ensure_agent_save_files("../evil")
    assert not (resources_root / "data").exists()


def test_failure_is_best_effort(resources_root, monkeypatch):
    def _boom(**kwargs):
        raise OSError("disk on fire")

    monkeypatch.setattr(startup, "_ensure_json_file", _boom)
    startup.ensure_agent_save_files("cf7_abc")  # 不抛，仅记日志
