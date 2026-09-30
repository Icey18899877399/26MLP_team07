# Twelve-model visualization implementation plan

**Goal:** Merge the current teammate backend and expose all twelve existing algorithm families through a real training and visualization frontend.

**Architecture:** Vue calls the current native HTTP contract: model IDs include the variant, results contain metrics and metadata. ML adapters execute the repository's scratch Models classes and generate JSON-safe visualization options from the actual run. Preserve model implementations and existing repository data. Teacher exemplar video (MLTaskConfig, 13:54) recovered and visually inspected: configuration, logs/status, metrics, enlargable charts and history guide the UI.

**Tech stack:** Vue 3, ECharts, FastAPI, ml_core, Python; retain explicit dependency metadata.

## Shared interfaces

- GET /api/models returns ModelInfo: id, display_name, task, compatible_datasets, default_params, parameter_descriptions.
- GET /api/datasets returns DatasetInfo: id, display_name, task, sample_count, feature_count, has_target.
- POST /api/experiments accepts model, dataset, params, test_size, random_state. No variant field.
- ExperimentResult: run_id, model, dataset, task, effective_params, metrics, artifacts, metadata.
- metadata.visualizations: list of {id, title, description, option}. option is a JSON-safe ECharts option (no callbacks). Charts describe actual results and label projections/sampling clearly.
- metadata.training_history and metadata protocol details can be extended without HTTP schema changes. Do not fabricate epoch progress. Preserve queued/running/result/error UI states.

## Independent work areas

- [x] Backend sync: imported upstream f33e37c source/tests and native HTTP docs, retained monorepo dependency path. Native requests and legacy variant rejection covered; 39 backend tests passed before model expansion.
- [ ] Model adapters: register all 12 families with actual scratch estimators, dataset resources/loaders, deterministic valid train/test evaluation, true per-task charts. Add all-family smoke tests and update discovery assertions. Keep Models unchanged.
- [ ] Frontend: consume native contract, task/family catalog, meaningful parameter controls, training status, result charts, data overview and comparable run history. Add native request/result unit tests and extend browser smoke.
- [ ] Integration: install declared dependencies; run backend, model and frontend tests, build frontend; exercise all 12 algorithms through HTTP and representative charts through browser. Review changes and fix actionable findings.
- [ ] Delivery: document upstream SHA, supported algorithms and datasets, sampling/normalization semantics and limits. Commit and push the validated result to the user's main with a non-force update.

## Acceptance checks

Catalog contains all 12 model families and no unavailable options. Every advertised model's default configuration runs on a compatible packaged dataset. Results serialize with allow_nan=False, metrics are finite, visualizations contain actual numeric series, and training errors are visible. Regression preprocessing is fitted on training data only. Anomaly evaluation uses held-out labels; clustering projections are explicitly labeled. Browser settings/history migrate safely to the new native contract. No fake curves, placeholder model results or unavailable cancellation claims.
