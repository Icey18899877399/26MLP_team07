# Original Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Restore the user's original 12 experiment workflows and 72 figures in a Geeker-inspired frontend.

**Architecture:** Add a separate original-experiment public catalog and subprocess runner to ml_core. FastAPI manages persisted serial background jobs and safe artifact downloads. Vue consumes this API exclusively for the new primary pages, clearly separating archived figures from new outputs.

**Tech Stack:** Python 3.12, FastAPI, existing scratch models/Matplotlib, Vue 3, Pinia, Element Plus, Vite.

**Spec:** docs/superpowers/specs/2026-09-30-original-experiments-design.md

## Global Constraints

- Do not edit Models/, data/, visualization/, figures/.
- Original scripts are the authority; no quick mode, arbitrary sample caps, replacement datasets or reconstructed charts.
- Existing figures are archived outputs; reruns write new unique directories only.
- No demo data, simulated metrics or fake progress.
- Existing public single-fit API remains compatible but is not the primary UI workflow.
- Backend imports original functionality through public ml_core exports only.
- Work in the user's existing checkout (main changes/push were explicitly authorized); preserve unrelated work. No worker pushes or commits until controller verifies combined results.

## Shared HTTP contract

`GET /api/original-experiments` -> array of `{id,title,task,script,datasets:[{id,name,path}],parameters:{},protocol:[string],figures:[{name,title,url}],source_data:[{name,url}]}`. IDs are bare family names (e.g. logistic_regression); task values classification/regression/clustering/anomaly_detection.

`GET /api/original-datasets` -> array of `{id,name,paths:[string],used_by:[string],archived_only:boolean}`. Include original Adult and California files as archived-only, not enabled replacement inputs.

`POST /api/original-runs` body `{experiment_id:string}` -> 202 job; `GET /api/original-runs` -> array newest first; `GET /api/original-runs/{run_id}` -> job. Job: `{run_id,experiment_id,status,created_at,started_at,finished_at,parameters,figures,source_data,log,error}`. Status queued/running/completed/failed/interrupted. Times ISO strings/null. log string; error string/null; assets match catalog descriptor format. Unknown IDs 404, invalid body 422, capacity 409.

All artifact URLs start with `/api/`; frontend prefixes configured backend origin for image/download requests. GET `/api/original-assets/{experiment_id}/{filename:path}` serves catalog-whitelisted original files. GET `/api/original-runs/{run_id}/assets/{filename:path}` serves only that run's PNG/CSV files below its unique output folder. No arbitrary filesystem exposure.

### Task 1: Original catalog, runner, and HTTP jobs

**Files:** new ml_core/original*.py (split catalog/execution), public exports in ml_core/__init__.py; backend/backend/interaction/original_routes.py and backend/backend/integration/original_jobs.py; register router/lifecycle in backend/backend/main.py; root pyproject.toml packaging/dependencies; tests/test_original_experiments.py and backend/tests/test_original_experiments.py; .gitignore runtime directory.

**Interfaces:** public `list_original_experiments()`, `list_original_datasets()`, and runner/root resolver as needed (document exact exports). HTTP produces shared contract above.

- [x] Write failing tests: catalog 12 entries/6 existing images each, logical grouping and data sources; unknown ID/path traversal; POST is nonblocking; failed subprocess doesn't return success; stale job recovery.

```python
def test_original_gallery_has_all_figures(client):
    response = client.get('/api/original-experiments')
    assert response.status_code == 200
    entries = response.json()
    assert len(entries) == 12
    assert sum(len(item['figures']) for item in entries) == 72
    image = client.get(entries[0]['figures'][0]['url'])
    assert image.content.startswith(b'\x89PNG')
```

- [x] Run `.venv/Scripts/python -m pytest tests/test_original_experiments.py backend/tests/test_original_experiments.py` and record RED.
- [x] Build catalog from original scripts/data, explicit allowlist of 12 script modules; original roots resolution supports editable checkout and wheel. Invoke CLI with only original dataset and fresh output args, e.g. `[sys.executable, '-m', 'visualization.logistic_regression_figures', '--data', original_data, '--output', unique_output]`. No test/seed/iteration overrides.
- [x] Persist job metadata, serialize queue, capture log tail, mark failure/interruption accurately, validate paths, keep output dirs gitignored. Ensure normal service shutdown handles workers/child processes and restart never leaves perpetual running state.
- [x] Run focused tests GREEN, add real logistic full workflow smoke plus installed-wheel catalog check if practical. Supply implementation report and test evidence.

### Task 2: Geeker-inspired original experiment UI

**Files:** frontend/src/App.vue, styles/main.css, router/index.js; replace primary ExperimentView.vue, DatasetsView.vue, BenchmarkView.vue (history), ManualView.vue; new services/originalApi.js, stores/originalExperiments.js, components for original figure gallery/protocol/job panel; frontend/tests/original*.test.js.

**Interfaces:** consumes shared HTTP contract; reuse httpClient configured origin; no calls to legacy POST /api/experiments from primary pages. No hardcoded dataset/model catalogs masquerading as backend availability.

- [x] First write behavioral tests for source/run distinction, API URLs/payload, terminal-state polling and endpoint changes; run `npm --prefix frontend test` and record RED.

```javascript
test('run request cannot carry peer defaults', async () => {
  const calls = []
  const api = createOriginalApi({post: async (...args) => {calls.push(args); return {run_id:'r1'}}})
  await api.startRun('logistic_regression')
  assert.deepEqual(calls[0], ['/api/original-runs', {experiment_id:'logistic_regression'}])
})
```

- [x] Create foldable navigation, breadcrumb/header tabs, restrained teal accent and white cards on neutral gray; reference Geeker layout, not its mock charts. Desktop gallery 2 columns with readable uncropped images, mobile single column.
- [x] Render catalog's actual PNGs and optional CSV downloads; protocol shows original settings read-only with explanation. Label original archived outputs vs rerun outputs prominently; no old generic chart gallery on primary path.
- [x] Start/poll real job with logs and elapsed time (no guessed percentage); prevent duplicate clicks and surface API failures. Provide persistent server history page and resume polling after reload. Clear stale selection/status when backend origin changes; unavailable service must not enable run.
- [x] Update datasets/manual using original API; show actual-used vs archived datasets, explain cross-dataset experiments and original synthetic mechanism demos. Run tests GREEN and build; supply report.

### Task 3: Integration verification, provenance documentation, delivery

**Files:** README.md, docs/FULL_STACK.md, docs/SOURCES.md, this plan progress; integration test/evidence files outside original source trees.

- [x] Review task diffs for spec compliance and correctness, resolve issues before completion.
- [x] Run backend tests and frontend tests/build; launch localhost app; use browser to verify 12 algorithms, all 72 image requests, modal/download, mobile navigation, real full logistic run, archived-vs-new distinction, failures and history refresh.
- [x] Verify `git diff -- Models data visualization figures` is empty; original asset hashes unchanged. Do not claim all 12 full computational pipelines were rerun unless they actually finished.
- [x] Document new endpoints, pip dependencies and wheel asset layout, job persistence/restart semantics, no new arbitrary data caps; preserve old API as legacy compatibility only.
Delivery: commit scoped verified changes and push main as already authorized; verify remote SHA and report precise limitations. Delivery commit IDs are recorded in Git history.

## Progress

- Planning: user approved architecture; source audit confirms 72 identical PNGs, original code/data identical ignoring CRLF.
- Tasks 1–3 implementation and verification: complete. See docs/ORIGINAL_VERIFICATION.md for evidence and limits.
- Task reviews and final cross-layer review: approved after frontend race/deep-link fixes and backend recovered-log fix.
- Worktree policy: existing main checkout used under prior explicit main-update authorization; no original source directories changed.
