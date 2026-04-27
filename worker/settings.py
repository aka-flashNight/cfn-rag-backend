"""Worker 配置。

arq 为可选依赖（server profile），仅在需要时导入 RedisSettings。
"""

from __future__ import annotations

from core.config import get_settings


def _get_broker_url() -> str:
    settings = get_settings()
    return settings.worker_broker_url


def create_worker_settings(**override_kwargs) -> "arq.connections.RedisSettings":
    """创建 arq RedisSettings 实例（延迟导入 arq）。"""
    from arq.connections import RedisSettings

    url = _get_broker_url()
    kwargs = dict(host="localhost", port=6379, database=1)
    parsed = _parse_redis_url(url)
    kwargs.update(parsed)
    kwargs.update(override_kwargs)
    return RedisSettings(**kwargs)


def _parse_redis_url(url: str) -> dict:
    """简易 Redis URL 解析。"""
    result: dict = {}
    try:
        # 格式: redis://[:password@]host:port/db
        without_scheme = url.replace("redis://", "").replace("rediss://", "")
        if "@" in without_scheme:
            auth, rest = without_scheme.split("@", 1)
            if ":" in auth:
                _, pwd = auth.split(":", 1)
                result["password"] = pwd
            without_scheme = rest
        if "/" in without_scheme:
            host_port, db_str = without_scheme.rsplit("/", 1)
            try:
                result["database"] = int(db_str)
            except ValueError:
                pass
        else:
            host_port = without_scheme
        if ":" in host_port:
            host, port_str = host_port.rsplit(":", 1)
            result["host"] = host
            try:
                result["port"] = int(port_str)
            except ValueError:
                pass
        else:
            result["host"] = host_port
    except Exception:
        pass
    return result
