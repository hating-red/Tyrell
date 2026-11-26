from types import SimpleNamespace

import pytest

import paired_research
import solution


@pytest.mark.parametrize(("outcomes", "expected"), [
    (["saved"], ["lightgbm"]),
    (["crash", "saved"], ["lightgbm", "xgboost"]),
    (["timeout", "not_saved", "saved"], ["lightgbm", "xgboost", "catboost"]),
    (["exception", "crash", "crash"], ["lightgbm", "xgboost", "catboost"]),
])
def test_retry_only_unpublished_initial_model(tmp_path, monkeypatch, outcomes, expected):
    clock = [1000.]
    deadline = clock[0] + 570
    calls, controls, searches = [], [], []
    accepted = {"saved": True, "fit_time": 1.}
    monkeypatch.setattr(solution, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(solution, "TOTAL_SECONDS", 570.)
    monkeypatch.delenv("TYRELL_PAIR_ARM", raising=False)
    monkeypatch.delenv("TYRELL_DIAGNOSTIC_ABLATIONS", raising=False)

    def run(task_dir, spec, budget, recipe="single", config=None, *, deadline):
        if recipe == "auto":
            return None, {"exit_code": 86}
        backend = (config or {}).get("model", "lightgbm")
        assert 1 <= budget <= deadline - clock[0]
        assert deadline <= 1540.  # preserve the final reserve
        outcome = outcomes[len(calls)]
        calls.append(backend)
        if outcome == "timeout":
            clock[0] = deadline
            return None, {"error": "worker timeout"}
        if outcome == "exception":
            raise OSError("worker could not start")
        if outcome == "not_saved":
            return {"saved": False}, {}
        if outcome == "crash":
            return None, {"exit_code": 86}
        return accepted, {}

    monkeypatch.setattr(solution, "experiment", run)
    monkeypatch.setattr(solution, "choose_variant", lambda path, spec, base, deadline: controls.append(base))
    monkeypatch.setattr(solution, "checkpoint_history", lambda path: [])
    monkeypatch.setattr(solution, "search_remaining_budget", lambda *args: searches.append("extra"))
    monkeypatch.setattr(paired_research, "run_paired_search", lambda *args: searches.append("paired"))
    solution.research(tmp_path, {}, "", {}, None, deadline)

    assert calls == expected
    assert controls == [accepted if expected == ["lightgbm"] else None]
    assert searches == ["paired", "extra"]
