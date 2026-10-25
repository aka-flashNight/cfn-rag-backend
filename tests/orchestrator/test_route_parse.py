"""route_search 参数扩容后的路由解析口径（keywords / sources / deep_search / query）。"""

from __future__ import annotations

from services.orchestrator.turn import _parse_route
from tests.orchestrator.conftest import tc


def _route_args(args: dict, name: str = "route_search"):
    return _parse_route([tc(name, args)])


def test_parse_full_search_params():
    route = _route_args({
        "keywords": "安迪 过去",
        "sources": ["knowledge", "items"],
        "deep_search": True,
        "query": "安迪与A兵团的往事",
    })
    assert route is not None
    assert route.kind == "search"
    assert route.keywords == "安迪 过去"
    assert route.sources == ("knowledge", "items")
    assert route.deep_search is True
    assert route.query == "安迪与A兵团的往事"


def test_parse_sources_string_and_all():
    assert _route_args({"keywords": "x", "sources": "stages"}).sources == ("stages",)
    assert _route_args({"keywords": "x", "sources": "all"}).sources == ("all",)
    assert _route_args({"keywords": "x"}).sources == ()


def test_deep_search_bool_coercion():
    assert _route_args({"keywords": "x", "deep_search": "true"}).deep_search is True
    assert _route_args({"keywords": "x", "deep_search": "false"}).deep_search is False
    assert _route_args({"keywords": "x", "deep_search": 1}).deep_search is True
    assert _route_args({"keywords": "x", "deep_search": 0}).deep_search is False


def test_parse_keywords_truncated_and_broken_json():
    route = _route_args({"keywords": "甲" * 200})
    assert len(route.keywords) == 80

    broken = _parse_route([{
        "id": "c0",
        "type": "function",
        "function": {"name": "route_search", "arguments": "{不是 JSON"},
    }])
    assert broken is not None
    assert broken.kind == "search"
    assert broken.keywords == ""
    assert broken.deep_search is False


def test_first_route_call_wins_and_non_route_ignored():
    calls = [
        tc("update_long_term_memory", {}, call_id="m0"),
        tc("route_search", {"keywords": "第一个"}, call_id="r1"),
        tc("route_task_draft", {"note": "第二个"}, call_id="r2"),
    ]
    route = _parse_route(calls)
    assert route is not None and route.kind == "search"
    assert route.keywords == "第一个"


def test_no_route_tool_returns_none():
    assert _parse_route([tc("update_long_term_memory", {})]) is None
    assert _parse_route([]) is None


def test_arguments_as_dict_not_string():
    route = _parse_route([{
        "id": "c0",
        "type": "function",
        "function": {"name": "route_search", "arguments": {"keywords": "安迪"}},
    }])
    assert route.keywords == "安迪"
