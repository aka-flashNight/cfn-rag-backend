#!/usr/bin/env python
"""
arq Worker 启动入口。

用法::

    python worker/main.py

依赖：
- Redis 已启动（作为 broker）
- 项目 .env 中配置了 CFN_WORKER_BROKER_URL
"""

if __name__ == "__main__":
    from worker import main
    main()
