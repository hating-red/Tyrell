import pytest

import solution


def anchor():
    return {"recipe": "single", "complete": True, "saved": True, "score": .15,
            "fit_time": 4.0, "config": {"params": {"num_leaves": 15}, "entity_history": True},
            "auto_features": True, "target_encoding": False, "diagnostic_group": None,
            "feature_code": None, "problems": []}


@pytest.mark.parametrize("limit,already_spent", [(600, 180), (7200, 1295), (7200, 159)])
def test_remaining_budget_fills_original_deadline_with_real_work(monkeypatch, limit, already_spent):
    now = [float(already_spent)]
    deadline = solution.training_seconds(limit) - 2.
    monkeypatch.setattr(solution.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(solution, "checkpoint_history", lambda _: [anchor()])
    attempts = []
    blends = []

    def fit(folder, spec, budget, config, recipe, feature_code, deadline):
        assert recipe == "bag" and feature_code is None
        assert config["entity_history"] and config["auto_features"] and "params" not in config
        assert budget > 0 and now[0] + budget <= deadline
        attempts.append((config["model"], config["seed"]))
        now[0] += budget + 1.
        return {"complete": True, "fit_time": budget, "score": .16, "members": {"count": 3}}, {}

    def blend(folder, spec, deadline):
        blends.append(now[0])
        now[0] = min(now[0] + 2., deadline)

    monkeypatch.setattr(solution, "experiment", fit)
    monkeypatch.setattr(solution, "ensemble", blend)
    solution.search_remaining_budget(".", {"metric": "pr_auc"}, deadline)
    assert {m for m, _ in attempts} == {"lightgbm", "xgboost", "catboost"}
    assert len({seed for _, seed in attempts}) == len(attempts)
    assert attempts and blends and 0 <= deadline - now[0] < 25


def test_backend_failures_stop_without_spinning(monkeypatch):
    called = []
    monkeypatch.setattr(solution, "checkpoint_history", lambda _: [anchor()])
    def fail(*args, **kwargs):
        called.append(kwargs["config"]["model"])
        return None, {"error": "backend unavailable"}
    monkeypatch.setattr(solution, "experiment", fail)
    solution.search_remaining_budget(".", {"metric": "mae"}, solution.time.monotonic() + 400)
    assert called == ["lightgbm", "xgboost", "catboost"]


def test_late_search_keeps_early_file_if_no_complete_control_or_time(monkeypatch):
    monkeypatch.setattr(solution, "checkpoint_history", lambda _: [anchor()])
    monkeypatch.setattr(solution, "experiment", lambda *a, **k: pytest.fail("no time"))
    solution.search_remaining_budget(".", {"metric": "mae"}, solution.time.monotonic() + 2)
    monkeypatch.setattr(solution, "checkpoint_history", lambda _: [anchor() | {"complete": False}])
    solution.search_remaining_budget(".", {"metric": "mae"}, solution.time.monotonic() + 500)
