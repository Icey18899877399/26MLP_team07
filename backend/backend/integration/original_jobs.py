"""Persistent, single-worker subprocess queue for original experiment scripts."""

from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from ml_core import build_original_command, list_original_experiments, list_original_run_assets

if os.name == "nt":
    from .windows_process_job import WindowsProcessJob


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class OriginalJobManager:
    def __init__(
        self,
        state_dir: Path,
        *,
        capacity: int = 8,
        command_builder: Callable[[str, Path], list[str]] = build_original_command,
    ) -> None:
        self.state_dir = Path(state_dir).resolve()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.capacity = capacity
        self.command_builder = command_builder
        self._lock = threading.RLock()
        self._queue: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self._process: subprocess.Popen | None = None
        self._process_job: WindowsProcessJob | None = None if os.name == "nt" else None
        self._jobs: dict[str, dict] = {}
        self._recover()

    def _job_dir(self, run_id: str) -> Path:
        return self.state_dir / run_id

    def _save(self, job: dict) -> None:
        directory = self._job_dir(job["run_id"])
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / "job.json"
        temporary = directory / "job.json.tmp"
        temporary.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(target)

    def _recover(self) -> None:
        for path in self.state_dir.glob("*/job.json"):
            if len(path.parent.name) != 32:
                continue
            try:
                uuid.UUID(hex=path.parent.name)
                job = json.loads(path.read_text(encoding="utf-8"))
                if job.get("run_id") != path.parent.name:
                    continue
                if job["status"] in {"queued", "running"}:
                    job["status"] = "interrupted"
                    job["finished_at"] = _now()
                    job["error"] = "服务重启，未完成的实验已中断"
                    job["log"] = self._read_log(job["run_id"])
                    self._save(job)
                self._jobs[job["run_id"]] = job
            except (OSError, ValueError, KeyError, TypeError):
                continue

    def submit(self, experiment_id: str) -> dict:
        catalog = {item["id"]: item for item in list_original_experiments()}
        if experiment_id not in catalog:
            raise KeyError(experiment_id)
        with self._lock:
            if self._stop.is_set():
                raise RuntimeError("Job service is shutting down")
            active = sum(job["status"] in {"queued", "running"} for job in self._jobs.values())
            if active >= self.capacity:
                raise OverflowError("Original run queue is full")
            run_id = uuid.uuid4().hex
            job = {
                "run_id": run_id,
                "experiment_id": experiment_id,
                "status": "queued",
                "created_at": _now(),
                "started_at": None,
                "finished_at": None,
                "parameters": catalog[experiment_id]["parameters"],
                "figures": [],
                "source_data": [],
                "log": "",
                "error": None,
            }
            self._save(job)
            self._jobs[run_id] = job
            self._queue.put(run_id)
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._work, name="original-experiment-worker", daemon=True)
                self._worker.start()
            return dict(job)

    def _work(self) -> None:
        while not self._stop.is_set():
            try:
                run_id = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._run(run_id)
            finally:
                self._queue.task_done()

    def _run(self, run_id: str) -> None:
        with self._lock:
            job = self._jobs[run_id]
            if job["status"] != "queued" or self._stop.is_set():
                return
            job["status"] = "running"
            job["started_at"] = _now()
            self._save(job)
        output_dir = self._job_dir(run_id) / "output"
        log_file = self._job_dir(run_id) / "process.log"
        try:
            output_dir.mkdir(parents=True, exist_ok=False)
            command = self.command_builder(job["experiment_id"], output_dir)
            if self._stop.is_set():
                raise RuntimeError("Job service stopped before subprocess launch")
            with log_file.open("wb") as stream:
                child_env = os.environ.copy()
                child_env.update(PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
                process = subprocess.Popen(
                    command,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    env=child_env,
                    creationflags=(getattr(subprocess, "CREATE_NO_WINDOW", 0) | 0x00000004) if os.name == "nt" else 0,
                    start_new_session=os.name != "nt",
                )
                with self._lock:
                    self._process = process
                    if os.name == "nt":
                        self._process_job = WindowsProcessJob(process._handle)
                        if not self._stop.is_set():
                            WindowsProcessJob.resume(process._handle)
                    if self._stop.is_set():
                        self._terminate_tree(process)
                exit_code = process.wait()
            with self._lock:
                if self._process_job is not None:
                    self._process_job.close()
                    self._process_job = None
                self._process = None
                job["log"] = self._read_log(run_id)
                if self._stop.is_set():
                    job["status"] = "interrupted"
                    job["error"] = "服务停止，实验已中断"
                elif exit_code == 0:
                    figures, source_data = list_original_run_assets(job["experiment_id"], output_dir, run_id)
                    if len(figures) != 6:
                        job["status"] = "failed"
                        job["error"] = f"原脚本完成但只生成 {len(figures)} 张 PNG，预期 6 张"
                    else:
                        job["status"] = "completed"
                        job["figures"] = figures
                        job["source_data"] = source_data
                else:
                    job["status"] = "failed"
                    job["error"] = f"原脚本退出码 {exit_code}"
                job["finished_at"] = _now()
                self._save(job)
        except Exception as exc:
            with self._lock:
                if self._process is not None and self._process.poll() is None:
                    self._process.kill()
                    self._process.wait(timeout=5)
                if self._process_job is not None:
                    self._process_job.close()
                    self._process_job = None
                self._process = None
                job["status"] = "interrupted" if self._stop.is_set() else "failed"
                job["error"] = f"{type(exc).__name__}: {exc}"
                job["log"] = self._read_log(run_id)
                job["finished_at"] = _now()
                self._save(job)

    def _read_log(self, run_id: str) -> str:
        path = self._job_dir(run_id) / "process.log"
        if not path.exists():
            return ""
        with path.open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 16000))
            return stream.read().decode("utf-8", errors="replace")

    def get(self, run_id: str) -> dict:
        with self._lock:
            if run_id not in self._jobs:
                raise KeyError(run_id)
            job = dict(self._jobs[run_id])
            if job["status"] == "running":
                job["log"] = self._read_log(run_id)
            return job

    def list(self) -> list[dict]:
        with self._lock:
            run_ids = sorted(self._jobs, key=lambda key: self._jobs[key]["created_at"], reverse=True)
            return [self.get(run_id) for run_id in run_ids]

    def close(self) -> None:
        self._stop.set()
        with self._lock:
            process = self._process
            if process is not None and process.poll() is None:
                self._terminate_tree(process)
        if self._worker is not None:
            self._worker.join(timeout=5)
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                self._terminate_tree(self._process)
                if self._process.poll() is None:
                    self._process.kill()
                    self._process.wait(timeout=5)
            for job in self._jobs.values():
                if job["status"] in {"queued", "running"}:
                    job["status"] = "interrupted"
                    job["finished_at"] = _now()
                    job["error"] = "服务停止，实验已中断"
                    job["log"] = self._read_log(job["run_id"])
                    self._save(job)

    def _terminate_tree(self, process: subprocess.Popen) -> None:
        if process.poll() is not None:
            return
        if os.name == "nt":
            if self._process_job is not None:
                self._process_job.terminate()
                return
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                    check=False,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            except (OSError, subprocess.TimeoutExpired):
                pass
        else:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        if process.poll() is None:
            process.terminate()
