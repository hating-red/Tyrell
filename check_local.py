"""Run the competition ZIP against the two public databases using inherited PG credentials."""
import argparse
from contextlib import ExitStack
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

import numpy as np
import pandas as pd


def check(task, materials, code, root, seconds):
    case = f"task{task}"
    source = materials / case
    cwd = root / case
    cwd.mkdir()
    for filename in ("task.md", "sample_submission.csv"):
        shutil.copyfile(source / filename, cwd / filename)
    env = dict(os.environ, PGDATABASE=f"hack_final_task{task}", TYRELL_TIME_LIMIT=str(seconds))
    log = root / f"{case}.log"
    start = time.monotonic()
    with log.open("wb") as stream:
        result = subprocess.run([sys.executable, str(code), "task.md"], cwd=cwd, env=env,
                                stdout=stream, stderr=subprocess.STDOUT, timeout=seconds+10)
    messages = log.read_bytes().decode("utf-8", errors="replace")
    for line in messages.splitlines():
        if line.startswith(("JOIN ", "SUBMISSION_STATE=", "single ", "auto ", "ensemble ")):
            print(f"{case} {line}", flush=True)
    if result.returncode or "SUBMISSION_STATE=MODEL_SUCCESS" not in messages:
        raise RuntimeError(f"{case} did not finish with a trained model\n{messages[-3000:]}")
    names = {p.name for p in cwd.iterdir()}
    if names != {"task.md", "sample_submission.csv", "submission.csv"}:
        raise ValueError(f"Unexpected task directory contents: {names}")
    sample, out = [pd.read_csv(cwd / n, dtype={"event_id": str}, keep_default_na=False)
                   for n in ("sample_submission.csv", "submission.csv")]
    if not out.columns.equals(sample.columns) or not sample.event_id.equals(out.event_id) or not out.event_id.is_unique:
        raise ValueError(f"{case}: columns or ordered IDs differ from the template")
    target = "engaged" if task == 1 else "response_time_minutes"
    y = pd.to_numeric(out[target], errors="raise")
    if not np.isfinite(y).all() or y.min() < 0 or (task == 1 and y.max() > 1) or y.nunique() <= 1:
        raise ValueError(f"{case}: invalid or constant predictions")
    score = re.search(r"SUBMISSION_STATE=MODEL_SUCCESS score=([^\s]+)", messages)
    if not score or not np.isfinite(float(score[1])):
        raise ValueError(f"{case}: missing finite CV metric")
    hourly = ("weather", "traffic", "air_quality", "power_grid", "public_transport")
    expected = ("users", "ads", "campaigns") if task == 1 else ("districts", "weather", "traffic")
    for table in expected:
        joined = re.search(rf"^JOIN [^.\n]+\.{table}:.*$", messages, re.MULTILINE)
        if not joined or (table in hourly and "start+1h < " not in joined[0]):
            raise ValueError(f"{case}: missing required join for {table}")
    if task == 2:
        for table in hourly[2:]:
            entry = re.search(rf"^JOIN [^.\n]+\.{table}\b.*$", messages, re.MULTILINE)
            if not entry:
                print(f"{case} {table}: absent from join log; check available database tables")
            elif "start+1h < " not in entry[0]:
                raise ValueError(f"{case}: {entry[0]}")
    print(f"{case} rows={len(out)} distinct={y.nunique()} CV={score[1]} elapsed={time.monotonic()-start:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("materials", type=Path, help="directory containing task1/ and task2/")
    parser.add_argument("--archive", type=Path, help="test the exact ZIP; otherwise use adjacent solution.py")
    parser.add_argument("--work-dir", type=Path, help="new directory in which to retain logs and predictions")
    parser.add_argument("--time-limit", type=int, default=1200)
    parser.add_argument("--task", type=int, choices=(1, 2), help="run one public database")
    args = parser.parse_args()
    with ExitStack() as stack:
        if args.work_dir:
            root = args.work_dir.resolve()
            root.mkdir(parents=True, exist_ok=False)
        else:
            root = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        code = Path(__file__).resolve().with_name("solution.py")
        if args.archive:
            with zipfile.ZipFile(args.archive) as z:
                if any(Path(n).name != n for n in z.namelist()):
                    raise ValueError("Expected a flat competition ZIP")
                z.extractall(root / "code")
            code = root / "code" / "solution.py"
        for task in ([args.task] if args.task else [1, 2]):
            check(task, args.materials.resolve(), code, root, args.time_limit)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    main()
