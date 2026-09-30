# Dataset workbench and model-specific charts implementation plan

> **For agentic workers:** Use test-driven implementation with independent data-library and model-chart work; root owns frontend integration and final verification.

**Goal:** Preserve originals, remove requested page clutter, support dataset inspection/upload/real training, and provide distinct algorithm diagnostics.

**Architecture:** Add a local persistent data library and isolated tabular-training API. Reuse model classes and parameter schemas, not original output directories. Add diagnostics at fitted-model boundaries so graphs reflect the actual run. Vue dataset cards navigate to a schema/preview/training detail screen.

**Tech Stack:** Existing Python/FastAPI/NumPy and Vue/Pinia/Element Plus/ECharts; CSV/TSV via standard library, no new services.

**Spec:** User-approved chat design: upload → structure/preview → feature/target selection → compatible algorithms → parameter training → actual visualizations. Uploaded files persist locally, never replace originals or enter Git. Classification/regression require target, clustering/anomaly target optional; explicit validation, no silent row deletion.

## Global constraints

- Do not modify Models/, visualization/, data/, figures/; preserve 72 archived PNG.
- Original full-script reproduction remains independent of custom data training.
- CSV/TSV uploads with headers; reject malformed tables/duplicate headers/oversize input, safely store by generated ID, never execute uploaded content.
- Numerical selected features only for this scratch-model version. Report missing/nonfinite/categorical selected fields; no automatic deletion or imputation.
- Binary classification supported by current metric contract; reject other labels with clear reason. Optional anomaly labels use explicit 0 normal / 1 anomaly.
- Keep original tests/legacy APIs compatible where possible. Only actual model diagnostics; disclose projection/sample caps and unsupported charts.
- Use existing clean local main checkout as previously authorized; no unrelated worktree or dependency changes.

## Interface contract

- GET /api/data-library -> array of {id,name,source,row_count,column_count}; source is original or upload.
- GET /api/data-library/{id} -> same plus columns:[{name,dtype,missing_count,unique_count}], preview:[row objects], default_target:string|null, paths:[string].
- POST /api/data-library/uploads JSON {filename,content} -> detail; max UTF-8 content 5 MiB, 20000 rows, 128 columns. Header required.
- POST /api/data-library/compatibility JSON {dataset_id,feature_columns:[string],target_column:string|null,task} -> {models:[full model IDs],issues:[string]}.
- POST /api/data-library/experiments JSON same selection plus model,params,random_state,test_size(optional) -> current ExperimentResult with metadata.visualizations. Task derived from model, optional task accepted consistently with selection request if needed.
- Uploaded content stored in .uploaded-datasets/ (gitignored); original details use registered IDs only. No client-supplied filesystem paths.
- Model diagnostics helper consumes fitted model, actual train/test features and labels, predictions, feature/label names; returns JSON ECharts specifications with unique algorithm diagnostic IDs.

## Task 1 — Data library and real custom-data execution

Files: new backend/backend/integration/data_library.py, interaction/data_routes.py, ml_core/tabular.py, backend/tests/test_data_library.py; backend/main.py router/lifecycle; .gitignore.

- [x] RED: request /api/data-library/wdbc, assert 569 rows and preview/field structure; upload CSV to tmp store, recreate app, assert same ID/content. Current endpoints must fail.
- [x] Implement registered-original readers (WDBC/Concrete/Seeds/NPZ/Adult/California) and header CSV/TSV parsing; bounded samples and field summaries.
- [x] RED: invalid extensions, traversal names, duplicate headers, ragged rows, oversized content; assert rejected without files outside store.
- [x] Implement generated-ID atomic local persistence and validation, no original writes.
- [x] RED: upload numeric x1,x2,label dataset, select task classification and feature columns, train KNN, assert actual result.dataset equals upload ID and charts show chosen labels, not breast-cancer names.
- [x] Implement compatibility and real train using current model classes; test all four tasks, optional unlabeled metrics, invalid features/target, no train/test leakage and no silent caps.
- [x] Run focused Python tests; report exact contracts and limits for integration.

## Task 2 — Algorithm-specific truthful visualizations

Files: new ml_core/adapters/model_charts.py, existing classification/regression/clustering adapters and interactive anomaly path; tests/test_model_diagnostics.py.

- [x] RED: run each of 12 models; assert a distinct model-specific chart ID and nonempty series linked to actual fitted quantities.
- [x] Implement logistic coefficients/probabilities, KNN neighborhoods, Gaussian NB class distributions, CART tree, RF importance/tree diversity, linear coefficients/residuals, GBDT stage losses/importance, MLP architecture/loss, Kmeans centers/distances, DBSCAN core/border/noise and k-distance, IF score ranking, OCSVM margin/support diagnostics using attributes actually available.
- [x] Parameterize generic axes/labels for custom datasets; no hardcoded WDBC or Seeds labels for uploaded tables. Preserve legacy IDs where existing consumers depend on them.
- [x] Run original and custom fits, check finite JSON and actual sample counts. Do not generate extra training runs just to draw an invented diagnostic.

## Task 3 — Frontend cleanup and dataset workflow

Files: ExperimentView.vue, DatasetsView.vue, router/index.js, new DatasetDetailView.vue, services/dataLibraryApi.js, stores/dataLibrary.js, components/datasets/*, frontend tests and browser scripts.

- [x] RED browser check requested top intro, algorithm six-count badges, archive warning absent; original cards remain six.
- [x] Remove only requested intro/count/warning nodes; preserve sidebar and original galleries.
- [x] RED unit tests data detail request race and source-switch result invalidation; RED browser upload CSV opens persisted detail with schema and sample table.
- [x] Implement data cards/detail navigation, upload picker with visible validation/limits, task/target/feature/model selection; reuse HyperparamForm and ChartGallery.
- [x] Disable incompatible training; bind submitted result to frozen selection, clear/label stale result when selection changes, show actual pending/error states.
- [x] Validate upload→refresh→details→two parameter runs in browser; original seven details and all twelve distinct graph signatures.

## Task 4 — Final verification and delivery

- [x] Independent read-only review of backend security/compatibility and frontend state handling; repair important findings with regression tests.
- [x] Root pytest, backend pytest separately, npm test/build and browser smoke.
- [x] Verify hashes for 148 original files unchanged; git check-ignore upload artifacts.
- [ ] Update user docs with binary/numeric limits and real verification evidence. Commit scoped code and push authorized main, verify remote SHA.
