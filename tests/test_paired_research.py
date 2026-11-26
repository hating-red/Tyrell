import importlib.util
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from paired_research import QWEN_ENV, compare_arms, run_paired_search

ROOT = Path(__file__).resolve().parents[1]


def prepare(tmp_path):
    (tmp_path / "train.csv").write_text("id,x,label\n1,private_row,1\n", encoding="utf-8")
    (tmp_path / "test.csv").write_text("id,x\n2,private_row\n", encoding="utf-8")
    (tmp_path / "sample_submission.csv").write_text("id,label\n2,0\n", encoding="utf-8")
    (tmp_path / "task.md").write_text("Private task conditions", encoding="utf-8")
    history = [{"recipe": "single", "complete": True, "score": .2, "fold_scores": [.1, .3],
                "training_seconds": 1, "saved": True, "config": {"entity_history": True, "categorical_ids": True}}]
    (tmp_path / ".tyrell_experiments.json").write_text(json.dumps(history), encoding="utf-8")
    return {"task_type": "binary", "metric": "pr_auc", "target_column": "label"}


def make_report(arm, search):
    return {"status": "ok", "csv_valid": True, "pair_arm": arm, "paired_search": search,
            "cv_audit": {"membership_signature": "folds", "folds": [{"model_score": .4}]},
            "metric": "pr_auc", "task": "t", "target": "y", "time_limit_seconds": 600,
            "budget_scope": "whole_diagnostic", "solution_budget_seconds": 574,
            "remaining_for_research_seconds": 545, "temporal_cv_score": .4}


def test_adaptive_pair_and_fail_closed_checks(tmp_path, monkeypatch):
    spec = prepare(tmp_path)
    monkeypatch.setenv("TYRELL_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("OPENAI_MODEL", "fixture-model")
    captured = []
    client = Mock()

    def completion(**kwargs):
        payload = json.loads(kwargs["messages"][1]["content"])
        assert "private_row" not in str(kwargs) and "Private task" not in str(kwargs)
        assert "target_column" not in payload and kwargs["model"] == "fixture-model"
        choice = list(payload["candidates"])[-1]
        backend = payload["candidates"][choice]["model"]
        override = {"num_leaves": 23} if backend == "lightgbm" else {"max_depth": 6}
        captured.append(choice)
        content = json.dumps({"hypothesis": choice, "params": override})
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    client.chat.completions.create.side_effect = completion
    experiment = Mock(return_value=({"score": .4, "fold_scores": [.4], "complete": True}, {}))
    monkeypatch.setenv("TYRELL_PAIR_ARM", "automl")
    before = run_paired_search(tmp_path, spec, None, time.monotonic() + 100, experiment)
    assert experiment.call_args_list[0].kwargs["config"]["entity_history"]
    frequency = experiment.call_args_list[2].kwargs["config"]
    assert frequency["cat_encoding"] == "frequency" and not frequency["categorical_ids"]
    monkeypatch.setenv("TYRELL_PAIR_ARM", "qwen")
    after = run_paired_search(tmp_path, spec, client, time.monotonic() + 100, experiment)
    assert len(captured) == len(set(captured)) == after["qwen_calls"] == 3
    assert after["trials"][0]["config"]["params"]["num_leaves"] == 23
    assert captured != [t["hypothesis"] for t in before["trials"]]
    assert before["protocol"] == after["protocol"]
    control, model = make_report("automl", before), make_report("qwen", after)
    assert compare_arms(control, model)["comparable"]
    for key in ("common_history", "inputs", "implementation", "candidates"):
        corrupt = after | {"protocol": after["protocol"] | {key: "different"}}
        assert compare_arms(control, make_report("qwen", corrupt))["gain"] is None
    for bad in (model | {"remaining_for_research_seconds": 475}, model | {"time_limit_seconds": 7200},
                model | {"cv_audit": {"membership_signature": "different"}},
                model | {"paired_search": after | {"qwen_calls": 0}},
                model | {"paired_search": after | {"phase_budget_seconds": 10}}):
        assert not compare_arms(control, bad)["comparable"]
    assert json.loads((tmp_path / "pair-search.json").read_text(encoding="utf-8")) == after


@pytest.mark.parametrize("reply", [
    {"code": "private_row"}, {"hypothesis": "shallow_xgb", "split": {"type": "random"}},
    {"hypothesis": "shallow_xgb", "params": {"model_file": "private_row"}},
    {"hypothesis": "shallow_xgb", "params": {"max_depth": 100}},
    {"hypothesis": "shallow_xgb", "params": {"eta": float("nan")}},
    {"hypothesis": "regularized_lgb", "params": {"num_leaves": True}},
])
def test_invalid_request_never_reaches_experiment(tmp_path, monkeypatch, reply):
    spec = prepare(tmp_path)
    monkeypatch.setenv("TYRELL_PAIR_ARM", "qwen")
    monkeypatch.setenv("OPENAI_MODEL", "fixture-model")
    monkeypatch.setenv("TYRELL_OUTPUT_DIR", str(tmp_path))
    client = Mock()
    message = SimpleNamespace(content=json.dumps(reply))
    client.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(message=message)])
    experiment = Mock()
    r = run_paired_search(tmp_path, spec, client, time.monotonic() + 100, experiment)
    assert (r["status"], r["qwen_calls"], r["attempted_calls"]) == ("qwen_request_failed", 0, 1)
    assert "private_row" not in (tmp_path / "pair-search.json").read_text(encoding="utf-8")
    experiment.assert_not_called()


@pytest.mark.parametrize("arm", [None, "automl", "qwen"])
def test_solver_environment_contains_only_opted_in_api_settings(tmp_path, monkeypatch, arm):
    spec = importlib.util.spec_from_file_location("diagnose", ROOT / "runs/real-public/diagnose.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in QWEN_ENV + ("OPENAI_LOG",):
        monkeypatch.setenv(name, "fixture")
    monkeypatch.setenv("TYRELL_PAIR_ARM", "wrong-inherited-arm")
    call = Mock(return_value=0)
    monkeypatch.setattr(module, "supervised_exit", call)
    assert module.run_solution(tmp_path, time.monotonic() + 100, None, arm, 600) == 0
    env = call.call_args.kwargs["env"]
    assert {k for k in env if k.startswith("OPENAI_")} == (set(QWEN_ENV) if arm == "qwen" else set())
    assert env.get("TYRELL_PAIR_ARM") == arm
    assert env["TYRELL_MAX_LLM_CALLS"] == ("3" if arm == "qwen" else "0")
