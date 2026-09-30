# Twelve-model visualization implementation plan

**Goal:** Merge the current teammate backend and expose all twelve existing algorithm families through a real training and visualization frontend.

**Architecture:** Vue calls the current native HTTP contract: model IDs include the variant, results contain metrics and metadata. ML adapters execute the repository's scratch Models classes and generate JSON-safe visualization options from the actual run. Preserve model implementations and existing repository data. Teacher exemplar video (MLTaskConfig, 13:54) recovered and visually inspected: configuration, logs/status, metrics, enlargable charts and history guide the UI.

**Tech stack:** Vue 3, ECharts, FastAPI, ml_core, Python; retain explicit dependency metadata.

**2026-09-30 scope update:** User supplied three teammate PRs and requested synchronization before continuing. Use their canonical twelve optimized model IDs and six datasets (not the earlier proposed 24 variants). Preserve local checkpoint 391b637; integrate ML PR a72ea7f, backend PR e83f128 and frontend PR 2100a6e. Keep native HTTP, add genuine charts and explicit anomaly in-sample evaluation labels. Do not mutate teammates' remote repositories.

## Shared interfaces

- GET /api/models returns ModelInfo: id, display_name, task, compatible_datasets, default_params, parameter_descriptions.
- GET /api/datasets returns DatasetInfo: id, display_name, task, sample_count, feature_count, has_target.
- POST /api/experiments accepts model, dataset, params, test_size, random_state. No variant field.
- ExperimentResult: run_id, model, dataset, task, effective_params, metrics, artifacts, metadata.
- metadata.visualizations: list of {id, title, description, option}. option is a JSON-safe ECharts option (no callbacks). Charts describe actual results and label projections/sampling clearly.
- metadata.training_history and metadata protocol details can be extended without HTTP schema changes. Do not fabricate epoch progress. Preserve queued/running/result/error UI states.

## Independent work areas

- [x] Backend sync: imported upstream f33e37c source/tests and native HTTP docs, retained monorepo dependency path. Native requests and legacy variant rejection covered; 39 backend tests passed before model expansion.
- [x] Model adapters: merged peer twelve optimized runners and six datasets; added genuine per-task charts and strict finite JSON. Classified/regression hold-outs remain disjoint, anomaly is explicitly sample-internal. Corrected positive-class ROC probability column ordering. Models unchanged; 358 unittest tests plus focused chart checks passed.
- [x] Frontend: integrated peer native HTTP, Chinese workbench and all four views, true chart gallery/enlargement/PNG export, history/reproduction/comparison. Fixed endpoint races, numeric gamma, quota handling and obsolete mock history isolation; 14 tests passed.
- [x] Integration: 358 model unittest tests, 5 focused chart tests, 56 backend tests, 14 frontend tests, production build and isolated wheel installation passed. All 12 algorithms passed both real HTTP smoke and browser smoke; browser also covered chart/image/JSON export, persisted history, six datasets, comparison, manual and 390px layout. Independent code review has no remaining blockers.
- [x] Delivery preparation: recorded all three upstream PR SHAs, supported algorithms/datasets, sampling and evaluation limitations. Validated integration is ready for the user-authorized non-force main update; remote success is verified in the task handoff.

## Acceptance checks

Catalog contains all 12 model families and no unavailable options. Every advertised model's default configuration runs on a compatible packaged dataset. Results serialize with allow_nan=False, metrics are finite, visualizations contain actual numeric series, and training errors are visible. Regression preprocessing is fitted on training data only. Anomaly evaluation uses held-out labels; clustering projections are explicitly labeled. Browser settings/history migrate safely to the new native contract. No fake curves, placeholder model results or unavailable cancellation claims.
