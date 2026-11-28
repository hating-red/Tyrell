import pytest

import pipeline
import solution


def anchor():
    return {"recipe": "single", "complete": True, "saved": True, "score": .15, "greater_is_better": True,
            "fit_time": 4.0, "fold_signature": "paired", "fold_scores": [.15] * 5,
            "config": {"model": "lightgbm", "auto_features": True}}


@pytest.mark.parametrize("limit,already_spent", [(600, 180), (7200, 1295), (7200, 159)])
def test_structured_search_never_uses_the_refit_reserve(monkeypatch, limit, already_spent):
    now = [float(already_spent)]
    deadline = limit - 2.
    monkeypatch.setattr(solution.time, "monotonic", lambda: now[0])
    control = anchor()
    monkeypatch.setattr(pipeline, "experiment_history", lambda _: [control])
    attempts = []
    settings = {"metric": "pr_auc", "available_backends": ["lightgbm", "xgboost", "catboost"]}

    def fit(folder, spec, fit_deadline, budget, config, recipe="single", *, estimate):
        assert fit_deadline <= deadline - 400
        if recipe == "tune":
            return {"tuned_params": {}}
        attempts.append(config["model"])
        now[0] += estimate
        return control

    monkeypatch.setattr(solution, "run_trial", fit)
    search = solution.search_remaining_budget(".", settings, deadline, control, {}, set())
    assert now[0] <= deadline - 400
    assert search["attempts"] == len(attempts)
    if limit == 7200:
        assert attempts and search["reason"] in {"queue_exhausted", "stagnation"}


def test_failed_candidates_stop_without_spinning(monkeypatch):
    called = []
    control = anchor()
    monkeypatch.setattr(pipeline, "experiment_history", lambda _: [control])

    def fail(folder, spec, deadline, budget, config, recipe="single", **extra):
        if recipe == "single":
            called.append(config["model"])
        return None

    monkeypatch.setattr(solution, "run_trial", fail)
    result = solution.search_remaining_budget(".", {"metric": "mae"}, solution.time.monotonic() + 2000,
                                              control, {}, set())
    assert result["reason"] == "stagnation" and len(called) == 8
    assert set(called) == {"lightgbm", "xgboost", "catboost"}


def test_late_search_keeps_early_file_if_no_time(tmp_path, monkeypatch):
    output = tmp_path / "submission.csv"
    output.write_text("id,y\\nx,1\\n", encoding="utf-8")
    before = output.read_bytes()
    control = anchor()
    monkeypatch.setattr(pipeline, "experiment_history", lambda _: [control])
    monkeypatch.setattr(solution, "run_trial", lambda *a, **k: pytest.fail("no time"))
    result = solution.search_remaining_budget(tmp_path, {"metric": "mae"}, solution.time.monotonic() + 402,
                                              control, {}, set())
    assert result["attempts"] == 0 and output.read_bytes() == before


def test_long_variant_queue_checkpoints_before_auto_training(tmp_path, monkeypatch):
    import training
    from test_final_agent import CASES, write_fixture

    task, _ = write_fixture(tmp_path, CASES[0])
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.delenv("PGDATABASE", raising=False)
    monkeypatch.setenv("TYRELL_LLM", "mock")
    now = [0.]
    monkeypatch.setattr(solution.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(training, "backend_available", lambda name, **_: name == "lightgbm")
    monkeypatch.setattr(solution, "choose_variant", lambda *args, **kwargs: [{"seed": i} for i in range(1, 5)])
    monkeypatch.setattr(solution, "_loss_variants", lambda *args, **kwargs: [])
    monkeypatch.setattr(solution, "search_remaining_budget", lambda *args, **kwargs: {"reason": "test"})
    history, attempts = [], []
    monkeypatch.setattr(pipeline, "experiment_history", lambda _: history)

    def trial(path, spec, deadline, budget, config=None, recipe="single", **kwargs):
        attempts.append((recipe, now[0]))
        if recipe == "single":
            now[0] += 300.
            report = anchor() | {"preds_file": "candidate.npz", "elapsed_seconds": 300.}
            history.append(report)
            return report
        return None

    monkeypatch.setattr(solution, "run_trial", trial)
    solution.research(task, work, 5000.)
    assert ("ensemble", 900.) in attempts
    assert attempts.index(("ensemble", 900.)) < next(i for i, (kind, _) in enumerate(attempts) if kind == "auto")
