import json
import sys
import time
import pytest

from fastapi.testclient import TestClient

from backend.main import create_app


def _wait_for(client, run_id, expected):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        job = client.get(f"/api/original-runs/{run_id}").json()
        if job["status"] == expected:
            return job
        time.sleep(0.02)
    raise AssertionError(f"job never reached {expected}: {job}")


def test_original_gallery_serves_all_original_images(tmp_path):
    with TestClient(create_app(original_jobs_dir=tmp_path)) as client:
        response = client.get("/api/original-experiments")
        assert response.status_code == 200
        entries = response.json()
        assert len(entries) == 12
        assert sum(len(item["figures"]) for item in entries) == 72
        image = client.get(entries[0]["figures"][0]["url"])
        assert image.status_code == 200
        assert image.content.startswith(b"\x89PNG")
        download = client.get(entries[0]["figures"][0]["url"] + "?download=true")
        assert download.headers["content-disposition"].startswith("attachment;")
        assert client.get("/api/original-assets/unknown/01.png").status_code == 404
        assert client.get("/api/original-assets/logistic_regression/not-real.png").status_code == 404


def test_post_is_nonblocking_and_rejects_unknown_or_extra_fields(tmp_path):
    with TestClient(create_app(original_jobs_dir=tmp_path)) as client:
        assert client.post("/api/original-runs", json={"experiment_id": "unknown"}).status_code == 404
        assert client.post("/api/original-runs", json={"experiment_id": "logistic_regression", "seed": 1}).status_code == 422


def test_failed_subprocess_is_persisted_as_failed(tmp_path):
    from backend.integration.original_jobs import OriginalJobManager

    manager = OriginalJobManager(
        tmp_path,
        command_builder=lambda _id, _dir: [sys.executable, "-c", "import sys; print('expected failure'); sys.exit(7)"],
    )
    with TestClient(create_app(original_job_manager=manager)) as client:
        started = time.monotonic()
        response = client.post("/api/original-runs", json={"experiment_id": "logistic_regression"})
        assert response.status_code == 202
        assert time.monotonic() - started < 1
        run_id = response.json()["run_id"]
        job = _wait_for(client, run_id, "failed")
        assert "expected failure" in job["log"]
        assert job["error"]
        assert job["figures"] == []
        assert client.get("/api/original-runs").json()[0]["run_id"] == run_id
        assert json.loads((tmp_path / run_id / "job.json").read_text(encoding="utf-8"))["status"] == "failed"


def test_restart_marks_unfinished_jobs_interrupted(tmp_path):
    from backend.integration.original_jobs import OriginalJobManager

    run_id = "a" * 32
    folder = tmp_path / run_id
    folder.mkdir()
    (folder / "job.json").write_text(json.dumps({
        "run_id": run_id, "experiment_id": "logistic_regression", "status": "running",
        "created_at": "2026-01-01T00:00:00+00:00", "started_at": "2026-01-01T00:00:01+00:00",
        "finished_at": None, "parameters": {}, "figures": [], "source_data": [], "log": "", "error": None,
    }), encoding="utf-8")
    (folder / "process.log").write_text("partial output before crash\n", encoding="utf-8")
    manager = OriginalJobManager(tmp_path)
    job = manager.get(run_id)
    assert job["status"] == "interrupted"
    assert job["finished_at"]
    assert job["error"]
    assert "partial output before crash" in job["log"]
    with TestClient(create_app(original_job_manager=manager)) as client:
        assert client.get(f"/api/original-runs/{run_id}").json()["log"] == job["log"]
    persisted = json.loads((folder / "job.json").read_text(encoding="utf-8"))
    assert persisted["log"] == job["log"]


def test_queue_capacity_rejects_additional_job_without_losing_existing_job(tmp_path):
    from backend.integration.original_jobs import OriginalJobManager

    manager = OriginalJobManager(
        tmp_path,
        capacity=1,
        command_builder=lambda _id, _dir: [sys.executable, "-c", "import time; time.sleep(10)"],
    )
    try:
        with TestClient(create_app(original_job_manager=manager)) as client:
            first = client.post("/api/original-runs", json={"experiment_id": "logistic_regression"})
            second = client.post("/api/original-runs", json={"experiment_id": "knn"})
            assert first.status_code == 202
            assert second.status_code == 409
            assert len(client.get("/api/original-runs").json()) == 1
    finally:
        manager.close()


def test_shutdown_interrupts_running_subprocess(tmp_path):
    from backend.integration.original_jobs import OriginalJobManager

    manager = OriginalJobManager(
        tmp_path,
        command_builder=lambda _id, _dir: [sys.executable, "-c", "import time; time.sleep(10)"],
    )
    started = time.monotonic()
    with TestClient(create_app(original_job_manager=manager)) as client:
        response = client.post("/api/original-runs", json={"experiment_id": "logistic_regression"})
        run_id = response.json()["run_id"]
        _wait_for(client, run_id, "running")
    assert manager.get(run_id)["status"] == "interrupted"
    assert manager.get(run_id)["finished_at"]
    assert time.monotonic() - started < 3


def test_run_asset_resolver_rejects_symlink_escape(tmp_path):
    from ml_core import resolve_original_run_asset
    import pytest

    root = tmp_path / "output"
    root.mkdir()
    outside = tmp_path / "secret.png"
    outside.write_bytes(b"secret")
    link = root / "secret.png"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("Windows symlink creation requires privilege")
    with pytest.raises(ValueError):
        resolve_original_run_asset(root, "secret.png")


def test_real_full_logistic_pipeline_generates_six_new_figures_and_csvs(tmp_path):
    with TestClient(create_app(original_jobs_dir=tmp_path)) as client:
        response = client.post("/api/original-runs", json={"experiment_id": "logistic_regression"})
        assert response.status_code == 202
        run_id = response.json()["run_id"]
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            job = client.get(f"/api/original-runs/{run_id}").json()
            if job["status"] in {"completed", "failed", "interrupted"}:
                break
            time.sleep(0.1)
        else:
            raise AssertionError("Full logistic workflow exceeded 240 seconds")
        assert job["status"] == "completed", job["log"][-2000:]
        assert len(job["figures"]) == 6
        assert len(job["source_data"]) == 6
        assert client.get(job["figures"][0]["url"]).content.startswith(b"\x89PNG")
        download = client.get(job["figures"][0]["url"] + "?download=true")
        assert download.headers["content-disposition"].startswith("attachment;")
        assert client.get(job["source_data"][0]["url"]).status_code == 200


def test_app_import_is_storage_free_and_default_history_is_cwd_independent(tmp_path, monkeypatch):
    from pathlib import Path

    monkeypatch.chdir(tmp_path)
    app = create_app()
    assert not (tmp_path / ".original-runs").exists()
    with TestClient(app) as client:
        assert client.get("/api/original-runs").status_code == 200
        expected = Path(__file__).resolve().parents[2] / ".original-runs"
        assert app.state.original_jobs.state_dir == expected.resolve()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows process-tree shutdown")
def test_shutdown_terminates_spawned_child_process_tree(tmp_path):
    from backend.integration.original_jobs import OriginalJobManager

    marker = tmp_path / "orphan.txt"
    child_code = f"import pathlib,time; time.sleep(3); pathlib.Path({str(marker)!r}).write_text('survived')"
    parent_code = (
        "import subprocess,sys,time; "
        f"subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
        "print('child-ready', flush=True); time.sleep(30)"
    )
    manager = OriginalJobManager(
        tmp_path / "runs",
        command_builder=lambda _id, _dir: [sys.executable, "-c", parent_code],
    )
    with TestClient(create_app(original_job_manager=manager)) as client:
        response = client.post("/api/original-runs", json={"experiment_id": "logistic_regression"})
        run_id = response.json()["run_id"]
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            job = client.get(f"/api/original-runs/{run_id}").json()
            if "child-ready" in job["log"]:
                break
            time.sleep(0.03)
        else:
            raise AssertionError(f"Child process never started: {job}")
    time.sleep(3.3)
    assert not marker.exists(), "a descendant of the stopped run kept executing"
