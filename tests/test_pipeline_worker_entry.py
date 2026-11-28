import csv
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

import pipeline
import solution
from publication import publication_run


@pytest.mark.parametrize("inherited", [None, "unrelated-inherited-directory"])
def test_pipeline_worker_preserves_import_environment(monkeypatch, tmp_path, inherited):
    if inherited is None:
        monkeypatch.delenv("PYTHONPATH", raising=False)
    else:
        monkeypatch.setenv("PYTHONPATH", inherited)
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-only")
    monkeypatch.setenv("PGPASSWORD", "synthetic-only")
    spec = {"task_type": "regression", "metric": "mae"}

    def inspect(command, deadline, **kwargs):
        assert command == [sys.executable, "-B", "-u", str(Path(solution.__file__).with_name("pipeline.py").resolve())]
        assert kwargs["env"].get("PYTHONPATH") == inherited
        assert not {"OPENAI_API_KEY", "PGPASSWORD"} & kwargs["env"].keys()
        assert kwargs["cwd"] == tmp_path
        payload = json.load(kwargs["stdin"])
        assert payload["task_spec"] == spec and payload["config"] == {"seed": 7}
        assert payload["recipe"] == "tune"
        return subprocess.CompletedProcess(command, 0, stdout=b"log\n{}\n", stderr=b"")

    monkeypatch.setattr(solution, "run_until", inspect)
    assert solution.run_trial(tmp_path, spec, time.monotonic() + 30, 10., {"seed": 7}, "tune") is not None


def test_real_workers_use_separate_code_data_and_output(monkeypatch, tmp_path):
    data, output = tmp_path / "snap shot", tmp_path / "task"
    data.mkdir()
    output.mkdir()
    monkeypatch.delenv("PYTHONPATH", raising=False)
    sink = pipeline.output_directory
    token = sink.set(str(output))
    (data / "pipeline.py").write_text("raise RuntimeError('wrong worker')", encoding="utf-8")
    spec = {"task_type": "regression", "metric": "mae", "target_column": "response",
            "id_columns": ["event_id"], "probabilities": False}
    with (data / "train.csv").open("w", newline="", encoding="utf-8") as file:
        w = csv.writer(file)
        w.writerow(["event_id", "feature", "response"])
        for n in range(300):
            w.writerow([str(n), n % 30, n % 30])
    ids = [str(n) for n in range(300, 330)]
    with (data / "test.csv").open("w", newline="", encoding="utf-8") as file:
        w = csv.writer(file)
        w.writerow(["event_id", "feature"])
        w.writerows((n, int(n) % 30) for n in ids)
    with (data / "sample_submission.csv").open("w", newline="", encoding="utf-8") as file:
        w = csv.writer(file)
        w.writerow(["event_id", "response"])
        w.writerows((n, -900) for n in reversed(ids))
    try:
        with publication_run():
            for seed in (12, 41):
                report = solution.run_trial(data, spec, time.monotonic() + 35, 20.,
                                            {"seed": seed, "params": {"num_leaves": 10, "min_child_samples": 4},
                                             "num_boost_round": 10})
                assert report and report["complete"]
            assert solution.run_trial(data, spec, time.monotonic() + 30, 15., recipe="ensemble")
    finally:
        sink.reset(token)
    history = json.loads((data / ".tyrell_experiments.json").read_text(encoding="utf-8"))
    assert [h["recipe"] for h in history] == ["single", "single", "ensemble"]
    assert all(h["complete"] for h in history)
    assert not (data / "submission.csv").exists()
    with (output / "submission.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert [r["event_id"] for r in rows] == ids[::-1]
    assert len({r["response"] for r in rows}) > 1
    assert all(float(r["response"]) >= 0 for r in rows)
