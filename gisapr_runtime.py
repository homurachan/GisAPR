"""Lightweight controller: exactly one long-lived search process per device.

This module intentionally never imports torch. GPU workers are initialized
sequentially; only their requests run concurrently (important on slow NFS).
"""
import atexit
import os
import secrets
import subprocess
import sys
import threading
import time
from multiprocessing.connection import Listener
from pathlib import Path

SEARCH_SCRIPT = Path(__file__).resolve().with_name(
    "test1_with_isspa_weight_varingKK_search_translation_also_v606_torch_optimized_standalone.py"
)
_workers = {}
_registry_lock = threading.RLock()


def resolve_script(path=None):
    if not path:
        return SEARCH_SCRIPT
    path = Path(path).expanduser()
    if path.is_file():
        return path.resolve()
    sibling = SEARCH_SCRIPT.parent / path.name
    if not path.is_absolute() and sibling.is_file():
        return sibling
    raise FileNotFoundError(f"Search script not found: {path}")


def device_ids(value):
    ids = [part.strip() for part in str(value).split(":") if part.strip()]
    if not ids or any(x != "cpu" and not x.isdigit() for x in ids):
        raise ValueError("--gpuid must be cpu or colon-separated logical CUDA IDs, e.g. 0:1")
    return list(dict.fromkeys(ids))


class PermanentWorker:
    def __init__(self, script, gpuid, startup_timeout=900):
        self.lock = threading.RLock()
        self.process = None
        self.connection = None
        self.gpuid = str(gpuid)
        authkey = secrets.token_bytes(32)
        listener = Listener(("127.0.0.1", 0), family="AF_INET", authkey=authkey)
        accepted = []
        error = []

        def accept():
            try:
                accepted.append(listener.accept())
            except Exception as exc:
                error.append(exc)

        thread = threading.Thread(target=accept, daemon=True)
        thread.start()
        host, port = listener.address
        command = [sys.executable, "-u", str(script), "--worker-connect", host,
                   str(port), "--worker-auth", authkey.hex(), "--worker-gpuid", self.gpuid]
        # Inherit CUDA_VISIBLE_DEVICES from SLURM; gpuid indexes that visible set.
        env = os.environ.copy()
        env.setdefault("OMP_NUM_THREADS", "4")
        env.setdefault("MKL_NUM_THREADS", "4")
        print(f"Starting permanent worker on {self.gpuid} ...", flush=True)
        try:
            self.process = subprocess.Popen(command, env=env)
            deadline = time.monotonic() + float(startup_timeout)
            while not accepted:
                if error:
                    raise RuntimeError(f"Worker connection failed: {error[0]}")
                if self.process.poll() is not None:
                    raise RuntimeError(f"Worker on {self.gpuid} exited with code {self.process.returncode}")
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Worker initialization exceeded {startup_timeout}s on {self.gpuid}")
                thread.join(0.1)
            self.connection = accepted[0]
            while not self.connection.poll(0.1):
                if self.process.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError(f"Worker on {self.gpuid} did not become ready")
            ready = self.connection.recv()
            if not ready.get("ok") or ready.get("event") != "ready":
                raise RuntimeError(f"Invalid worker greeting: {ready}")
            print(f"Permanent worker ready: device={self.gpuid}, pid={ready.get('pid')}", flush=True)
        except BaseException:
            self.close()
            raise
        finally:
            listener.close()

    def request(self, command, **payload):
        with self.lock:
            if self.process.poll() is not None:
                raise RuntimeError(f"Worker on {self.gpuid} exited with code {self.process.returncode}")
            self.connection.send({"command": command, "cwd": os.getcwd(), **payload})
            while not self.connection.poll(0.5):
                if self.process.poll() is not None:
                    raise RuntimeError(f"Worker on {self.gpuid} died during {command}")
            try:
                response = self.connection.recv()
            except EOFError as exc:
                raise RuntimeError(f"Worker on {self.gpuid} disconnected during {command}") from exc
            if not response.get("ok"):
                raise RuntimeError(f"Worker {self.gpuid}: {response.get('error')}\n{response.get('traceback', '')}")
            return response.get("result")

    def close(self):
        # Cleanup also works after partially completed initialization.
        if self.connection is not None:
            try:
                if self.process is not None and self.process.poll() is None:
                    self.connection.send({"command": "shutdown"})
            except (OSError, EOFError):
                pass
        if self.process is not None:
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        if self.connection is not None:
            self.connection.close()


def get_worker(gpuid, script=None, startup_timeout=900):
    script = resolve_script(script)
    key = (str(script), str(gpuid))
    with _registry_lock:
        if key not in _workers:
            _workers[key] = PermanentWorker(script, gpuid, startup_timeout)
        return _workers[key]


def prepare_workers(args):
    ids = device_ids(args.gpuid)
    if getattr(args, "doSplitDiffGpu", False) and len(ids) == 1 and ids[0] != "cpu":
        ids = [str(int(ids[0]) + i) for i in range(max(1, int(args.SplitParticles)))]
    try:
        for gpuid in ids:
            get_worker(gpuid, getattr(args, "search_script", None), getattr(args, "worker_startup_timeout", 900))
    except BaseException:
        shutdown_workers()
        raise
    return ids


def shutdown_workers(args=None):
    with _registry_lock:
        workers = list(_workers.values())
        _workers.clear()
    for worker in workers:
        worker.close()


atexit.register(shutdown_workers)
