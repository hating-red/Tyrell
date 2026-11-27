"""Wall-clock supervision shared by the official CLI and the local diagnostics."""
import math
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
    return min(30.0, limit / 240)


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


def supervised_exit(command, deadline, **popen_options):
    try:
        return run_until(command, deadline, **popen_options).returncode
    except subprocess.TimeoutExpired:
        print("Wall-clock deadline reached; stopped the worker tree.", file=sys.stderr, flush=True)
        return 124
