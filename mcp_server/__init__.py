# CFN-RAG MCP Server — 游戏数据查询与创作辅助
#
# 【当前状态：代码保留，未启用】
# 9679ffe（v3/P0）曾将本目录随 server profile 一并删除，但 MCP 与 server profile
# 无关（走 stdio，不依赖 Redis/Postgres/Qdrant），属误伤，故从 master 取回。
#
# 未启用的原因与后续方向：
# - main.py 不引用本包，exe 打包清单亦不含，故对本地 exe 用户零影响；
# - fastmcp 未列入 requirements.txt（开发期按需 pip install fastmcp 即可）；
# - 定位待定：当前主链路是本服务作为 MCP Client 调用游戏服务的 MCP 工具；
#   若未来游戏侧要接入 AI，可将本服务以 MCP Server 暴露，避免双方重复造轮子。
