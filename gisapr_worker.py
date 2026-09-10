"""Persistent, authenticated local worker for GisAPR refinement.

The optimized search program imports this module only for its worker mode.
Keeping the controller out of this process lets each GPU import PyTorch once;
geometry checks, PDB density generation and particle searches then reuse it.
The wire protocol uses standard-library multiprocessing connections and Python
dictionaries, and deliberately does not change any search result files.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import os
from pathlib import Path
import sys
import traceback
from multiprocessing.connection import Client


def create_worker_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-connect", nargs=2, metavar=("HOST", "PORT"), required=True)
    parser.add_argument("--worker-auth", required=True, help="Local connection authentication key in hexadecimal")
    parser.add_argument("--worker-gpuid", default="0", help="GPU index, or cpu")
    return parser


def _geometry_cache_key(kwargs):
    """Invalidate prepared geometry when either its model or settings change."""
    path = Path(kwargs["pdb_path"]).expanduser().resolve()
    stat = path.stat()
    kwargs["pdb_path"] = str(path)
    settings = tuple(sorted(kwargs.items()))
    return settings, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def run_worker(create_search_parser, run_search, choose_device, argv=None):
    """Connect to the controller, execute requests in order, and remain alive.

    Requests:
      geometry: geometry_kwargs, pose, optional pivot
      generate_mrc: mrc_kwargs
      search: argv (the existing standalone search argument list)
      ping / shutdown: no additional arguments

    Any request may supply cwd.  Each job starts in the worker's initial cwd
    unless an explicit cwd is supplied.  The assigned GPU overrides any
    per-request device, so a persistent process cannot switch GPUs.
    """
    import torch

    args = create_worker_parser().parse_args(argv)
    host, port = args.worker_connect
    if host not in ("127.0.0.1", "localhost"):
        raise ValueError("The GisAPR worker must connect to a loopback host")
    authkey = bytes.fromhex(args.worker_auth)
    if not authkey:
        raise ValueError("Worker authentication key cannot be empty")
    gpu = str(args.worker_gpuid).strip().lower()
    if gpu != "cpu":
        gpu = int(gpu)
        if gpu < 0:
            raise ValueError("worker-gpuid must be a nonnegative GPU index or cpu")
    device = choose_device(gpu)
    search_gpuid = device.index if device.type == "cuda" else "cpu"
    initial_cwd = os.getcwd()
    geometry_cache = OrderedDict()
    geometry_cache_limit = 2

    with Client((host, int(port)), family="AF_INET", authkey=authkey) as connection:
        connection.send({"ok": True, "event": "ready", "pid": os.getpid(), "device": str(device)})
        while True:
            try:
                request = connection.recv()
            except (EOFError, ConnectionError):
                break

            try:
                if not isinstance(request, dict):
                    raise TypeError("Worker request must be a dictionary")
                command = request.get("command")
                os.chdir(request.get("cwd") or initial_cwd)
                if command == "shutdown":
                    connection.send({"ok": True, "result": None})
                    break
                with torch.inference_mode():
                    if command == "geometry":
                        from func_check_boundary_for_testing_v8 import GeometryRestraint

                        kwargs = dict(request["geometry_kwargs"])
                        kwargs["device"] = str(device)
                        key = _geometry_cache_key(kwargs)
                        geometry = geometry_cache.get(key)
                        if geometry is None:
                            geometry = GeometryRestraint(**kwargs)
                            geometry_cache[key] = geometry
                            # Pixel-size scans can change settings repeatedly;
                            # keep prepared model state bounded in a long run.
                            if len(geometry_cache) > geometry_cache_limit:
                                geometry_cache.popitem(last=False)
                        else:
                            geometry_cache.move_to_end(key)
                        overlap, distance = geometry.evaluate(request["pose"], pivot=request.get("pivot"))
                        result = (int(overlap), float(distance))
                    elif command == "generate_mrc":
                        from pdb2mrc_gpu_ver_fp32_v3 import generate_mrc

                        kwargs = dict(request["mrc_kwargs"])
                        kwargs["device"] = str(device)
                        result = generate_mrc(**kwargs)
                    elif command == "search":
                        search_args = create_search_parser().parse_args(request["argv"])
                        search_args.gpuid = search_gpuid
                        run_search(search_args)
                        result = None
                    elif command == "ping":
                        result = {"pid": os.getpid(), "device": str(device), "geometry_cache_entries": len(geometry_cache)}
                    else:
                        raise ValueError(f"Unknown worker command: {command!r}")
                response = {"ok": True, "result": result}
            except (Exception, SystemExit) as error:
                # argparse raises SystemExit on invalid job arguments.  Report
                # that job's failure without discarding the warmed-up worker.
                response = {
                    "ok": False,
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                }
            try:
                sys.stdout.flush()
                sys.stderr.flush()
                connection.send(response)
            except (EOFError, ConnectionError, BrokenPipeError):
                break
