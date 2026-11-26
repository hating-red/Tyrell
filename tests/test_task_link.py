import os
import subprocess
import sys

import pandas as pd
import pytest

from postgres_loader import _local_sample
from test_final_agent import CASES, ROOT, write_fixture


def link_input(path, source):
    try:
        path.symlink_to(source)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 1314:
            pytest.skip("Windows file symlink privileges unavailable")
        raise


def test_adjacent_linked_template_precedes_nested_sample(tmp_path):
    work = tmp_path / "work"
    nested = work / "data"
    nested.mkdir(parents=True)
    external = tmp_path / "mounted.csv"
    external.write_text("id,y\nB,0\nA,0\n", encoding="utf-8")
    (nested / "sample_submission.csv").write_text("id,y\nA,0\nB,0\n", encoding="utf-8")
    adjacent = work / "sample_submission.csv"
    link_input(adjacent, external)
    assert _local_sample(work / "task.md", "See sample_submission.csv") == adjacent


def test_linked_task_keeps_output_and_id_order_in_working_directory(tmp_path):
    work = tmp_path / "work"
    source = tmp_path / "platform"
    work.mkdir()
    source.mkdir()
    task, expected = write_fixture(work, CASES[0])
    write_fixture(source, CASES[0])
    expected.iloc[::-1].to_csv(source / "sample_submission.csv", index=False)
    task.unlink()
    link_input(task, source / "task.md")
    original = (work / "sample_submission.csv").read_bytes()
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith(("PG", "TYRELL_", "OPENAI_"))}
    env.update(TYRELL_TIME_LIMIT="24", TYRELL_MAX_LLM_CALLS="0", PYTHONIOENCODING="utf-8")
    run = subprocess.run([sys.executable, str(ROOT / "solution.py"), "task.md"], cwd=work,
                         env=env, capture_output=True, text=True, encoding="utf-8", timeout=36)
    assert run.returncode == 0, run.stderr
    result = pd.read_csv(work / "submission.csv", dtype={"event_id": str})
    assert result["event_id"].tolist() == expected["event_id"].tolist()
    assert result["engaged"].nunique() > 1
    assert (work / "sample_submission.csv").read_bytes() == original
    assert not (source / "submission.csv").exists()
