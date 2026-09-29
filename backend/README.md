# FastAPI 后端（整合版）

安装、启动和验证请参阅 [完整平台说明](../docs/FULL_STACK.md)。
从仓库根目录安装 `python -m pip install -e . -e "./backend[test]"`，然后运行 `python scripts/dev.py`。
仅启动后端：在本目录运行 `python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000`。

后端依赖根目录安装的 ml_core，通过公共接口调用；不再使用 external/ml-core 子模块。
GET /api/health、/api/models、/api/datasets 与 POST /api/experiments 已接入真实模型。
在本目录执行 `python -m pytest` 可验证原 HTTP 合同及真实模型集成。

docs/ 内原合同文档描述上游独立仓库设计，已标记为历史资料；本次适配规则以完整平台说明和 /docs 为准。
原仓库说明保存在 [UPSTREAM_README](docs/UPSTREAM_README.md)，来源见 [SOURCES](../docs/SOURCES.md)。
