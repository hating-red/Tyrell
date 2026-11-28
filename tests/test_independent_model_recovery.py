import pytest

import pipeline
import solution
import training
from test_final_agent import CASES, assert_format, write_fixture


@pytest.mark.parametrize(("completed_on", "expected"), [
    ("lightgbm", ["lightgbm"]),
    ("catboost", ["lightgbm", "catboost"]),
    ("xgboost", ["lightgbm", "catboost", "xgboost"]),
    (None, ["lightgbm", "catboost", "xgboost"]),
])
def test_only_an_unsuccessful_safety_model_tries_the_next_backend(tmp_path, monkeypatch, completed_on, expected):
    task, sample = write_fixture(tmp_path, CASES[0])
    worker = tmp_path / "worker"
    worker.mkdir()
    monkeypatch.delenv("PGDATABASE", raising=False)
    monkeypatch.setenv("TYRELL_LLM", "mock")
    monkeypatch.setattr(solution.time, "monotonic", lambda: 1000.)
    monkeypatch.setattr(training, "backend_available", lambda name, **_: name in solution._BACKENDS)
    monkeypatch.setattr(pipeline, "experiment_history", lambda _: [])
    monkeypatch.setattr(solution, "_loss_variants", lambda *args: [])
    calls, controls = [], []

    def variants(spec, baseline, **kwargs):
        controls.append(baseline)
        return []

    def run(directory, spec, deadline, budget, config=None, recipe="single", **kwargs):
        assert deadline <= 1200
        if recipe == "single":
            assert_format(worker / "submission.csv", sample, "ads")
            backend = config["model"]
            calls.append(backend)
            if backend == completed_on:
                return {"saved": True, "complete": True, "fit_time": 2., "fold_scores": [.4] * 5}
        return None

    monkeypatch.setattr(solution, "choose_variant", variants)
    monkeypatch.setattr(solution, "run_trial", run)
    solution.research(task, worker, 1200.)
    assert calls == expected
    assert len(controls) == 1
    assert (controls[0] or {}).get("backend") == completed_on
