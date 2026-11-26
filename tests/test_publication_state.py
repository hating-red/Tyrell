import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from pipeline import write_fallback
from publication import MODEL_SUCCESS, TRAINED_FALLBACK, publication_run, published_result
from submission import make_submission


def test_real_zero_is_published_and_not_inherited(tmp_path, monkeypatch):
    monkeypatch.setenv("TYRELL_OUTPUT_DIR", str(tmp_path))
    pd.DataFrame({"id": ["a", "b"], "x": [1, 3], "y": [0, 0]}).to_csv(tmp_path/"train.csv", index=False)
    pd.DataFrame({"id": ["c", "d"], "x": [2, 4]}).to_csv(tmp_path/"test.csv", index=False)
    sample = tmp_path / "sample_submission.csv"
    pd.DataFrame({"id": ["d", "c"], "y": [0.0, 0.0]}).to_csv(sample, index=False)
    out = tmp_path / "submission.csv"
    shutil.copyfile(sample, out)
    spec = {"id_column": "id", "target_column": "y", "task_type": "regression", "metric": "mae"}
    with publication_run():
        assert published_result(out) is None
        assert write_fallback(tmp_path, spec) == []
        assert out.read_bytes() == sample.read_bytes()
        assert published_result(out)["state"] == TRAINED_FALLBACK
    with publication_run():
        assert published_result(out) is None


def test_failed_replace_preserves_proof_and_tamper_invalidates_it(tmp_path, monkeypatch):
    ids = pd.Series(["a", "b"], name="id")
    sample = tmp_path / "sample_submission.csv"
    pd.DataFrame({"id": ids, "y": 0.0}).to_csv(sample, index=False)
    out = tmp_path / "submission.csv"
    with publication_run():
        assert not make_submission([1.0, 2.0], ids, sample, out, publication_state=TRAINED_FALLBACK)
        old = published_result(out)
        def fail(*args):
            raise OSError("injected replacement failure")
        with monkeypatch.context() as patch:
            patch.setattr(os, "replace", fail)
            with pytest.raises(OSError, match="replacement failure"):
                make_submission([4.0, 3.0], ids, sample, out, publication_state=MODEL_SUCCESS, score=1.2)
        assert published_result(out) == old
        assert not make_submission([4.0, 3.0], ids, sample, out, publication_state=MODEL_SUCCESS, score=1.2)
        assert published_result(out)["state"] == MODEL_SUCCESS
        assert published_result(out)["score"] == 1.2
        timestamps = out.stat()
        out.write_bytes(out.read_bytes().replace(b"4.0", b"5.0"))
        os.utime(out, ns=(timestamps.st_atime_ns, timestamps.st_mtime_ns))
        assert published_result(out) is None


def test_cli_failure_before_training_publishes_schema_fallback(tmp_path):
    task = tmp_path / "task.md"
    task.write_text("Target: y. Evaluation metric: MAE. ID: id.", encoding="utf-8")
    sample = tmp_path / "sample_submission.csv"
    sample.write_text("id,y\na,0\n", encoding="utf-8")
    out = tmp_path / "submission.csv"
    shutil.copyfile(sample, out)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PG", "OPENAI_", "TYRELL_"))}
    result = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve().parents[1] / "solution.py"),
                             str(task)], env=dict(env, PYTHONIOENCODING="utf-8"), cwd=tmp_path,
                            capture_output=True, timeout=30)
    assert result.returncode == 0
    assert b"SUBMISSION_STATE=SCHEMA_FALLBACK" in result.stdout
    assert b"Traceback" in result.stderr
    assert b"Warning: constant submission" in result.stderr
    assert out.read_text(encoding="utf-8").splitlines() == ["id,y", "a,1.0"]
    assert {p.name for p in tmp_path.iterdir()} == {"task.md", "sample_submission.csv", "submission.csv"}
