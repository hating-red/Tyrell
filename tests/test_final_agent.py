"""Entry-point tests for the final format. Set TYRELL_TEST_PG_BIN to run a disposable database."""

import json
import os
from pathlib import Path
import secrets
import shutil
import socket
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pytest

import solution
from postgres_loader import PG_ENV


ROOT = Path(__file__).resolve().parents[1]
CASES = [
    ("ads", "engaged", "average_precision_score", "pr_auc"),
    ("rescue", "response_time_minutes", "MAE", "mae"),
    ("private", "quality_band", "macro F1", "macro_f1"),
]


def fixture_frames(case):
    name, target, metric, _ = case
    n = 150
    signal = (np.arange(n) * 7) % 19
    identifier = "record_key" if name == "private" else "event_id"
    data = pd.DataFrame({identifier: [f"{i:04}" for i in range(n)], "intensity": signal})
    if name != "private":
        time_column = "event_time" if name == "ads" else "call_time"
        data[time_column] = pd.date_range("2025-06-01", periods=n, freq="2h").astype(str)
        data["district_id" if name == "rescue" else "ad_id"] = 1 + signal % 3
    if name == "ads":
        data[target] = (signal > 8).astype(int)
    elif name == "rescue":
        data[target] = 1 + 0.75 * signal
    else:
        data[target] = np.select([signal < 7, signal < 13], ["low", "mid"], default="high")
    train = data.iloc[:132].copy()
    test = data.iloc[132:].drop(columns=[target]).reset_index(drop=True)
    answer = "decision" if name == "private" else target
    sample = test[[identifier]].sample(frac=1, random_state=19).reset_index(drop=True)
    sample[answer] = "low" if name == "private" else 0
    task = f"Target: {target}. Evaluation metric: {metric}. ID: {identifier}.\n"
    return train, test, sample, task


def write_fixture(folder, case):
    train, test, sample, task = fixture_frames(case)
    for frame, name in ((train, "train"), (test, "test"), (sample, "sample_submission")):
        frame.to_csv(folder / (name + ".csv"), index=False)
    path = folder / "task.md"
    path.write_text(task, encoding="utf-8")
    return path, sample


@pytest.fixture
def clean_env(monkeypatch):
    for name in [*PG_ENV, "OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_BASE_URL", "TYRELL_OUTPUT_DIR"]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TYRELL_MAX_LLM_CALLS", "0")
    monkeypatch.setattr(solution, "MAX_LLM_CALLS", 0)


def assert_format(path, sample, name):
    identifier = sample.columns[0]
    frame = pd.read_csv(path, dtype={identifier: str})
    assert frame.columns.tolist() == sample.columns.tolist()
    assert frame[identifier].tolist() == sample[identifier].tolist()
    pred = frame.iloc[:, 1]
    if name == "private":
        assert set(pred) <= {"low", "mid", "high"} and pred.notna().all()
    else:
        assert np.isfinite(pred).all() and pred.ge(0).all()
        if name == "ads":
            assert pred.le(1).all()
    return pred


@pytest.mark.parametrize("case", CASES)
def test_model_failure_keeps_early_answer(tmp_path, case, monkeypatch, clean_env):
    path, sample = write_fixture(tmp_path, case)
    shutil.copyfile(tmp_path / "sample_submission.csv", tmp_path / "submission.csv")
    monkeypatch.setenv("TYRELL_RUNTIME_STATS", str(tmp_path / "runtime.json"))

    def broken(*args, **kwargs):
        assert_format(path.with_name("submission.csv"), sample, case[0])
        raise RuntimeError("injected model failure")

    monkeypatch.setattr(solution, "experiment", broken)
    solution.run_agent(path)
    pred = assert_format(path.with_name("submission.csv"), sample, case[0])
    assert pred.nunique() == 1
    stats = json.loads((tmp_path / "runtime.json").read_text(encoding="utf-8"))
    assert stats["submission_state"] == "TRAINED_FALLBACK"
    if case[0] != "private":
        assert pred.ne(0).all()


def test_missing_credentials_disable_client_with_calls_enabled(tmp_path, monkeypatch, clean_env):
    import openai

    path, sample = write_fixture(tmp_path, CASES[0])
    monkeypatch.setattr(solution, "MAX_LLM_CALLS", 150)

    def unexpected_client(**kwargs):
        pytest.fail("Created a model client without credentials")

    monkeypatch.setattr(openai, "OpenAI", unexpected_client)
    monkeypatch.setattr(solution, "research", lambda *args: None)
    solution.run_agent(path)
    assert_format(path.with_name("submission.csv"), sample, "ads")


def test_short_budget():
    assert solution.training_seconds(600) == 570
    assert 0 < solution.training_seconds() < 7200
    assert solution.training_seconds(14400) == solution.training_seconds()


@pytest.mark.parametrize("limit", [600, 7200])
def test_deadline_does_not_restart_after_imports(monkeypatch, tmp_path, limit):
    monkeypatch.setenv("TYRELL_TIME_LIMIT", str(limit))
    monkeypatch.delenv("TYRELL_RUNTIME_STATS", raising=False)
    monkeypatch.setattr(solution, "_run_agent", lambda task, deadline: deadline)
    hard = time.monotonic() + limit - 20  # supervisor already spent 20 s before run_agent
    assert solution.run_agent(tmp_path, deadline=hard) == hard - (limit - solution.training_seconds(limit))


@pytest.mark.parametrize("limit", [600, 7200])
def test_cli_budget_on_temporal_csv(tmp_path, clean_env, limit):
    path, sample = write_fixture(tmp_path, CASES[0])
    env = dict(os.environ, TYRELL_TIME_LIMIT=str(limit), PYTHONIOENCODING="utf-8",
               TYRELL_RUNTIME_STATS=str(tmp_path / "runtime.json"))
    with (tmp_path / "run.log").open("w", encoding="utf-8") as log:
        proc = subprocess.run([sys.executable, "-B", str(ROOT / "solution.py"), str(path)], env=env,
                              cwd=tmp_path, stdout=log, stderr=subprocess.STDOUT, timeout=limit)
    assert proc.returncode == 0, (tmp_path / "run.log").read_text(encoding="utf-8")
    assert_format(tmp_path / "submission.csv", sample, "ads")
    stats = json.loads((tmp_path / "runtime.json").read_text(encoding="utf-8"))
    history = json.loads((tmp_path / ".tyrell_experiments.json").read_text(encoding="utf-8"))
    assert stats["solution_limit_seconds"] == limit
    assert 0 < stats["remaining_for_research_seconds"] < solution.training_seconds(limit)
    assert stats["phase"] == "complete"
    assert stats["submission_state"] == "MODEL_SUCCESS"
    assert any(h["complete"] and h["saved"] for h in history)


def test_watchdog_kills_model_tree_and_preserves_published_csv(tmp_path):
    import psutil
    from time_budget import supervised_exit

    output = tmp_path / "submission.csv"
    output.write_text("event_id,engaged\n001,0.5\n", encoding="utf-8")
    before = output.read_bytes()
    code = ("import pathlib, subprocess, sys, time; "
            "proc = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
            "pathlib.Path('child.pid').write_text(str(proc.pid)); time.sleep(60)")
    started = time.monotonic()
    assert supervised_exit([sys.executable, "-c", code], started + 2, cwd=tmp_path) == 124
    elapsed = time.monotonic() - started
    assert elapsed < 5
    child_pid = int((tmp_path / "child.pid").read_text())
    if psutil.pid_exists(child_pid):
        child = psutil.Process(child_pid)
        # On POSIX an orphan can briefly await reaping; Windows reports termination asynchronously.
        try:
            if child.status() != psutil.STATUS_ZOMBIE:
                child.wait(timeout=2)
        except psutil.NoSuchProcess:
            pass
    assert output.read_bytes() == before


def test_official_cli_keeps_early_submission_on_stuck_worker(tmp_path, clean_env):
    path, sample = write_fixture(tmp_path, CASES[0])
    # Inject a stall after the production fallback writer publishes submission.csv.
    (tmp_path / "sitecustomize.py").write_text(
        "import sys, time\n"
        "if '--worker-deadline' in sys.argv:\n"
        "    import pipeline\n"
        "    original = pipeline.write_fallback\n"
        "    def stalled(*args, **kwargs):\n"
        "        original(*args, **kwargs)\n"
        "        time.sleep(60)\n"
        "    pipeline.write_fallback = stalled\n", encoding="utf-8")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(tmp_path), str(ROOT)]),
               TYRELL_TIME_LIMIT="8", PYTHONIOENCODING="utf-8")
    started = time.monotonic()
    proc = subprocess.run([sys.executable, "-B", str(ROOT / "solution.py"), str(path)], env=env,
                          cwd=tmp_path, capture_output=True, timeout=12)
    assert proc.returncode == 124, proc.stderr
    assert b"SUBMISSION_STATE=TRAINED_FALLBACK" in proc.stdout
    assert time.monotonic() - started <= 8
    assert_format(tmp_path / "submission.csv", sample, "ads")


def test_diagnostic_watchdog_includes_import_and_export(tmp_path):
    # Stuck import before any PostgreSQL connection, model training or sample generation.
    (tmp_path / "psycopg.py").write_text("import time\ntime.sleep(60)\n", encoding="utf-8")
    path, _ = write_fixture(tmp_path, CASES[0])
    output = tmp_path / "diagnostics"
    env = dict(os.environ, PYTHONPATH=str(tmp_path), PYTHONIOENCODING="utf-8")
    started = time.monotonic()
    proc = subprocess.run([sys.executable, "-B", str(ROOT / "runs/real-public/diagnose.py"), str(path),
                           "--run-dir", str(output), "--seconds", "3"], env=env,
                          capture_output=True, timeout=10)
    assert proc.returncode == 124, proc.stderr
    assert time.monotonic() - started < 6
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "deadline_exceeded" and report["time_limit_seconds"] == 3


@pytest.fixture(scope="module")
def local_cluster(tmp_path_factory):
    """Start only a new loopback cluster. The fixture never uses operator PG credentials."""
    folder = os.environ.get("TYRELL_TEST_PG_BIN")
    initdb = str(Path(folder) / "initdb") if folder else shutil.which("initdb")
    if not initdb:
        pytest.skip("initdb unavailable; point TYRELL_TEST_PG_BIN at PostgreSQL bin")
    root = tmp_path_factory.mktemp("postgres")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PG", "OPENAI_"))}
    password = secrets.token_urlsafe(32)
    data = root / "cluster"
    from tempfile import NamedTemporaryFile
    with NamedTemporaryFile(mode="w", dir=root, encoding="utf-8", delete=False) as secret:
        secret.write(password + "\n")
    try:
        result = subprocess.run([initdb, "-D", str(data), "--no-locale", "-E", "UTF8", "-U",
                                 "fixture_admin", "--auth=scram-sha-256", "--pwfile", secret.name],
                                env=env, capture_output=True, timeout=60)
        assert result.returncode == 0, result.stderr.decode(errors="replace")
    finally:
        Path(secret.name).unlink()
    pg_ctl = str(Path(initdb).with_name("pg_ctl"))
    subprocess.run([pg_ctl, "-D", str(data), "-l", str(root / "server.log"), "-w", "start",
                    "-o", f"-h 127.0.0.1 -p {port}"], env=env, check=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL, timeout=60)
    try:
        yield dict(host="127.0.0.1", port=port, user="fixture_admin", password=password)
    finally:
        subprocess.run([pg_ctl, "-D", str(data), "-w", "stop", "-m", "fast"],
                       env=env, check=True, capture_output=True, timeout=60)


def setup_database(cluster, case, task_dir):
    import psycopg
    from psycopg import sql

    name = case[0]
    dbname = "fixture_" + name
    with psycopg.connect(**cluster, dbname="postgres", autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(dbname)))
    train, test, sample, task = fixture_frames(case)
    train_name, test_name = ("measurements", "queue") if name == "private" else ("train", "test")
    if name == "private":
        task += "Train table measurements. Test table queue. Output column decision.\n"
    relations = [(train_name, train), (test_name, test)]
    if name == "rescue":
        table_sample = sample.assign(row_ordinal=np.arange(len(sample))).sample(frac=1, random_state=7)
        relations.append(("sample_submission", table_sample))
    else:
        sample.to_csv(task_dir / "sample_submission.csv", index=False)
    if name == "rescue":
        weather = pd.DataFrame({"district_id": [1, 2, 3], "hour_start": ["2025-06-01 00:00"]*3,
                                "temperature": [1, 2, 3]})
        relations.append(("weather", weather))
    if name == "ads":
        relations.append(("ads", pd.DataFrame({"ad_id": [1, 2, 3], "size": [3, 1, 7]})))
    with psycopg.connect(**cluster, dbname=dbname) as conn:
        for table, frame in relations:
            columns = []
            for col in frame:
                kind = "integer" if pd.api.types.is_integer_dtype(frame[col]) else (
                    "double precision" if pd.api.types.is_float_dtype(frame[col]) else "text")
                columns.append(sql.SQL("{} {}").format(sql.Identifier(col), sql.SQL(kind)))
            conn.execute(sql.SQL("CREATE TABLE {} ({})").format(sql.Identifier(table), sql.SQL(",").join(columns)))
            with conn.cursor().copy(sql.SQL("COPY {} FROM STDIN").format(sql.Identifier(table))) as stream:
                for row in frame.itertuples(index=False, name=None):
                    stream.write_row(row)
        role = "reader_" + name
        # The fixture admin and reader are local to this throwaway cluster.
        conn.execute(sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(sql.Identifier(role),
                                                                        sql.Literal(cluster["password"])))
        conn.execute(sql.SQL("GRANT SELECT ON ALL TABLES IN SCHEMA public TO {}").format(sql.Identifier(role)))
    path = task_dir / "task.md"
    path.write_text(task, encoding="utf-8")
    return sample, {"PGHOST": cluster["host"], "PGPORT": str(cluster["port"]), "PGUSER": role,
                    "PGPASSWORD": cluster["password"], "PGDATABASE": dbname}


@pytest.mark.parametrize("case", CASES)
def test_postgres_cli(tmp_path, case, local_cluster, clean_env):
    sample, connection = setup_database(local_cluster, case, tmp_path)
    env = dict(os.environ, **connection, TYRELL_TIME_LIMIT="65", PYTHONIOENCODING="utf-8")
    if os.environ.get("TYRELL_TEST_PLATFORM_LIMITS") == "1":
        if case[0] == "ads":
            del env["TYRELL_TIME_LIMIT"]
        elif case[0] == "rescue":
            env["TYRELL_TIME_LIMIT"] = "600"
    timeout = int(env.get("TYRELL_TIME_LIMIT", "7200")) + 15
    executable = Path(os.environ.get("TYRELL_TEST_SOLUTION", ROOT / "solution.py")).resolve()
    started = time.monotonic()
    import psutil
    with (tmp_path / "run.log").open("w", encoding="utf-8") as output:
        with subprocess.Popen([sys.executable, str(executable), "task.md"], cwd=tmp_path, env=env,
                              stdout=output, stderr=subprocess.STDOUT) as proc:
            main = psutil.Process(proc.pid)
            peak = 0
            while proc.poll() is None:
                if time.monotonic() - started > timeout:
                    proc.kill()
                    pytest.fail("solution exceeded its time limit")
                children = [main, *main.children(recursive=True)]
                rss = 0
                for child in children:
                    try:
                        rss += child.memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
                peak = max(peak, rss)
                time.sleep(0.1)
    logs = (tmp_path / "run.log").read_text(encoding="utf-8")
    assert proc.returncode == 0, logs
    pred = assert_format(tmp_path / "submission.csv", sample, case[0])
    history = json.loads((tmp_path / ".tyrell_experiments.json").read_text(encoding="utf-8"))
    assert any(h["saved"] and h["complete"] for h in history), logs
    assert any(h.get("auto_features") for h in history) and len(history) >= 2, logs
    assert history[0]["metric"] == case[3]
    if case[0] != "private":
        assert history[0]["split"]["type"] == "time"
        assert "JOIN public." in logs
    print(f"{case[0]}: {time.monotonic() - started:.1f}s, peak RSS {peak/2**20:.1f} MiB, "
          f"{len(history)} experiments, {pred.nunique()} distinct predictions, {tmp_path}")
