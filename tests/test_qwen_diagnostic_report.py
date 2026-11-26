import importlib.util
import io
import json
import time
from pathlib import Path

import pytest


path = Path(__file__).resolve().parents[1] / "runs" / "real-public" / "diagnose.py"
module_spec = importlib.util.spec_from_file_location("tyrell_diagnostic_report", path)
diagnose = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(diagnose)


def test_verification_rejects_an_unrelated_model_with_the_same_score(tmp_path):
    journal = [{"complete": True, "recipe": "single", "score": 1.2, "fold_signature": "folds",
                "config": {"model": "lightgbm", "params": {"num_leaves": 15}}}]
    trial = {"tool": "run_experiment", "journal_index": 0, "complete": True,
             "recipe": "single", "config": {"params": {"num_leaves": 31}}, "score": 1.2}
    state = {"status": "ok", "responses": 1, "attempted_calls": 1,
             "tool_calls": 1, "completed_experiments": 1, "trials": [trial]}
    output = {}
    runtime = tmp_path / "runtime.json"
    runtime.write_text(json.dumps({"qwen_tool_run": state}), encoding="utf-8")
    assert not diagnose.verify_qwen_experiments(tmp_path, journal, output)
    assert output["qwen_tool_run"]["verified_journal_indices"] == []
    trial["config"]["params"]["num_leaves"] = 15
    runtime.write_text(json.dumps({"qwen_tool_run": state}), encoding="utf-8")
    assert diagnose.verify_qwen_experiments(tmp_path, journal, output)
    assert output["qwen_tool_run"]["verified_journal_indices"] == [0]


@pytest.mark.parametrize("enable", [False, True])
def test_only_explicit_native_mode_forwards_qwen_environment(tmp_path, monkeypatch, enable):
    names = ("OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_BASE_URL")
    for name in names:
        monkeypatch.setenv(name, "fixture-value")
    monkeypatch.setenv("OPENAI_LOG", "debug")
    monkeypatch.setenv("TYRELL_PAIR_ARM", "qwen")
    monkeypatch.setenv("TYRELL_QWEN_TOOLS", "1")
    captured = {}

    def child(command, deadline, **kwargs):
        captured.update(kwargs["env"])
        return 0

    monkeypatch.setattr(diagnose, "supervised_exit", child)
    assert diagnose.run_solution(tmp_path, time.monotonic()+600, io.StringIO(), qwen_tools=enable) == 0
    assert "OPENAI_LOG" not in captured and "TYRELL_PAIR_ARM" not in captured
    assert captured["TYRELL_MAX_LLM_CALLS"] == ("2" if enable else "0")
    assert (captured.get("TYRELL_QWEN_TOOLS") == "1") == enable
    assert all((name in captured) == enable for name in names)
