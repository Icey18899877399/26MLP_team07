# FastAPI 后端（整合版）

安装、启动和验证请参阅 [完整平台说明](../docs/FULL_STACK.md)。
从仓库根目录安装 `python -m pip install -e . -e "./backend[test]"`，然后运行 `python scripts/dev.py`。
仅启动后端：在本目录运行 `python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000`。

后端依赖根目录安装的 ml_core，通过公共接口调用；不再使用 external/ml-core 子模块。
GET /api/health、/api/models、/api/datasets 与 POST /api/experiments 已接入真实模型。
在本目录执行 `python -m pytest` 可验证原 HTTP 合同及真实模型集成。

docs/ 内合同已同步队友的原生公共 API 更新：使用完整模型 ID，不再传 variant；结果直接包含 run_id、metrics 和 metadata。
本仓库以根目录的 ml-core 作为本地可编辑依赖，不使用上游独立部署中的 external/ml-core 子模块。
原仓库说明保存在 [UPSTREAM_README](docs/UPSTREAM_README.md)，来源见 [SOURCES](../docs/SOURCES.md)。
