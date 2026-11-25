"""Wall-clock supervision shared by the official CLI and the local diagnostics."""
from contextlib import ExitStack
import math
import os
from pathlib import Path
import subprocess
import sys
import time


def time_limit(value=7200):
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("TYRELL_TIME_LIMIT must be a positive finite number")
    return min(7200.0, seconds)


def shutdown_reserve(limit):
    # Include interpreter startup/shutdown and process-tree termination, also in short smoke tests.
    return min(2.0, limit / 4, max(0.35, 0.01 * limit))


def worker_command(script, deadline, args):
    return [sys.executable, "-B", "-u", str(script), "--worker-deadline", str(deadline), *args]


def pop_worker_deadline():
    if sys.argv[1:2] == ["--worker-deadline"]:
        deadline = float(sys.argv[2])
        del sys.argv[1:3]
        return deadline
    return None


def run_until(command, deadline, **popen_options):
    """Include process startup in the timeout and terminate model descendants on expiry."""
    import psutil

    if time.monotonic() >= deadline:
        raise subprocess.TimeoutExpired(command, 0)
    with subprocess.Popen(command, **popen_options) as proc:
        try:
            stdout, stderr = proc.communicate(timeout=max(0, deadline - time.monotonic()))
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            # Freeze each parent before enumerating children so no new worker is missed.
            try:
                pending = [psutil.Process(proc.pid)]
            except psutil.NoSuchProcess:
                pending = []
            while pending:
                process = pending.pop()
                try:
                    process.suspend()
                    pending.extend(process.children())
                    process.kill()
                except psutil.NoSuchProcess:
                    pass
            raise
        return subprocess.CompletedProcess(command, proc.returncode, stdout, stderr)


def _bundled_openmp():
    """Locate a working wheel runtime if the Linux loader lacks the system name."""
    import ctypes
    try:
        ctypes.CDLL("libgomp.so.1")
        return None
    except OSError:
        pass

    from importlib import metadata
    for package in ("scikit-learn", "xgboost"):
        try:
            dist = metadata.distribution(package)
        except metadata.PackageNotFoundError:
            continue
        for entry in dist.files or ():
            if not entry.match(package.replace("-", "_") + ".libs/libgomp-*.so.1*"):
                continue
            runtime = Path(dist.locate_file(entry)).resolve()
            try:
                ctypes.CDLL(str(runtime))
            except OSError:
                continue
            return runtime
    return None


def supervised_exit(command, deadline, **popen_options):
    try:
        with ExitStack() as resources:
            if sys.platform == "linux" and (runtime := _bundled_openmp()):
                from tempfile import TemporaryDirectory

                # The dynamic loader reads LD_LIBRARY_PATH at process startup. Keep
                # the alias alive for this worker and all of its model subprocesses.
                directory = resources.enter_context(TemporaryDirectory(prefix="tyrell-openmp-"))
                (Path(directory) / "libgomp.so.1").symlink_to(runtime)
                env = popen_options.get("env")
                child_env = dict(os.environ if env is None else env)
                previous = child_env.get("LD_LIBRARY_PATH")
                child_env["LD_LIBRARY_PATH"] = directory + (os.pathsep + previous if previous else "")
                popen_options["env"] = child_env
            return run_until(command, deadline, **popen_options).returncode
    except subprocess.TimeoutExpired:
        print("Wall-clock deadline reached; stopped the worker and its descendants.",
              file=sys.stderr, flush=True)
        return 124
