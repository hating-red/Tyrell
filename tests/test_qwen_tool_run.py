"""Offline tests for the limited native-tool AutoResearch diagnostic."""
import csv
from datetime import datetime, timedelta
import json
import logging
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import solution


SPEC = {"task_type": "regression", "metric": "mae", "target_column": "delay",
        "id_columns": ["event_id"], "probabilities": False}
ARGS = {"recipe": "single", "config": {"model": "lightgbm", "params": {"num_leaves": 20}}, "time_budget": 30}


def response(name=None, args=None, text=""):
    calls = [SimpleNamespace(id="mock-call", function=SimpleNamespace(name=name, arguments=json.dumps(args)))] if name else []
    message = SimpleNamespace(tool_calls=calls, content=text)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)],
                           usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7))


@pytest.fixture
def local_run(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "synthetic-model")
    monkeypatch.setenv("TYRELL_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(solution, "MAX_LLM_CALLS", 150)  # diagnostic enforces its own two-attempt ceiling
    monkeypatch.setattr(solution, "LLM_USAGE", {"calls": 0, "prompt": 0, "completion": 0})
    monkeypatch.setattr(solution, "HARD_DEADLINE", [None])
    (tmp_path / ".tyrell_experiments.json").write_text("[]")
    for file, length, offset in (("train", 120, 0), ("test", 12, 120), ("sample_submission", 12, 120)):
        with (tmp_path / (file + ".csv")).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            if file == "sample_submission":
                writer.writerow(["event_id", "delay"])
                writer.writerows((str(i), 0) for i in range(offset+length-1, offset-1,-1))
            else:
                writer.writerow(["event_id", "timestamp", "load"] + (["delay"] if file == "train" else []))
                for i in range(offset, offset+length):
                    timestamp = (datetime(2025, 1, 1) + timedelta(hours=i)).isoformat()
                    writer.writerow([str(i), timestamp, i%9] + ([1+i%9] if file=="train" else []))
    solution.write_fallback(tmp_path, SPEC)
    return tmp_path


def drive(folder, create):
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    state = solution.diagnostic_usage()
    solution.run_llm_loop(client, "", {}, SPEC, folder, time.monotonic() + 100, tool_state=state)
    return state


def test_native_tool_runs_real_experiment_and_supplies_result(local_run):
    first_result = response("run_experiment", ARGS, "private-model-text")
    first_result.choices[0].message.reasoning_content = "private-tool-turn-metadata"
    invoke = Mock(side_effect=[first_result, response(text="done")])
    state = drive(local_run, invoke)
    assert state["status"] == "ok" and state["completed_experiments"] == 1
    assert (state["attempted_calls"], state["responses"], state["tool_calls"]) == (2, 2, 1)
    assert (state["prompt_tokens"], state["completion_tokens"]) == (22, 14)
    trial = state["trials"][0]
    journal = json.loads((local_run / ".tyrell_experiments.json").read_text(encoding="utf-8"))
    assert trial["complete"] and trial["config"] == ARGS["config"] and trial["journal_index"] == 0
    assert trial["score"] == journal[0]["score"] and journal[0]["complete"]
    solution.check_result(local_run, SPEC)
    first, second = (call.kwargs for call in invoke.call_args_list)
    assert first["model"] == "synthetic-model"
    assert [tool["function"]["name"] for tool in first["tools"]] == ["run_experiment"]
    assert first["max_tokens"] == 1024 and 0 < first["timeout"] <= 45
    assert first["parallel_tool_calls"] is False
    feedback = [m for m in second["messages"] if m["role"] == "tool"]
    assert len(feedback) == 1 and json.loads(feedback[0]["content"])["score"] == trial["score"]
    assert second["messages"][-2]["reasoning_content"] == "private-tool-turn-metadata"
    assert "private-model-text" not in json.dumps(state) and "private-tool-turn-metadata" not in json.dumps(state)


@pytest.mark.parametrize("bad", [response(text="run the model"), response("run_python", {"code": "private"}),
                                  response("run_experiment", ARGS | {"feature_code": "private"}),
                                  response("run_experiment", ARGS | {"config": {}}),
                                  response("run_experiment", ARGS | {"config": {"params": {}}}),
                                  response("run_experiment", ARGS | {"config": {"params": {"objective": "private"}}})])
def test_invalid_or_text_only_never_dispatches_or_claims_success(local_run, bad, monkeypatch):
    experiment = Mock()
    monkeypatch.setattr(solution, "experiment", experiment)
    early = (local_run / "submission.csv").read_bytes()
    state = drive(local_run, Mock(side_effect=[bad,bad]))
    assert state["status"] in ("text_only", "invalid_or_failed_tool_call")
    assert state["attempted_calls"] == 2 and state["completed_experiments"] == 0
    assert "private" not in json.dumps(state)
    experiment.assert_not_called()
    assert (local_run / "submission.csv").read_bytes() == early


def test_retries_count_towards_paid_call_limit_and_errors_are_sanitized(local_run, monkeypatch, capsys, caplog):
    monkeypatch.setattr(time, "sleep", lambda _: None)
    def failed(**kwargs):
        logging.getLogger("httpx").warning("fake-auth-canary")
        raise RuntimeError("fake-auth-canary")
    create = Mock(side_effect=failed)
    state = drive(local_run, create)
    assert create.call_count == state["attempted_calls"] == 2
    assert state["status"] == "api_error" and state["completed_experiments"] == state["responses"] == 0
    assert "fake-auth-canary" not in capsys.readouterr().err + caplog.text + json.dumps(state)


def test_journal_mismatch_is_not_tool_success(local_run, monkeypatch):
    monkeypatch.setattr(solution, "experiment", lambda *args, **kwargs: ({"complete": True, "score": 1.},{}))
    r = response("run_experiment", ARGS)
    state = drive(local_run, Mock(side_effect=[r,r]))
    assert state["status"] == "experiment_incomplete" and not state["completed_experiments"]
    assert all(t["journal_index"] is None for t in state["trials"])


@pytest.mark.parametrize("mode", ["tools", "api_error", "malformed_spec"])
def test_official_entry_runs_bounded_autoresearch_without_mode_flag(local_run, monkeypatch, mode):
    import openai
    import postgres_loader
    import shutil
    from contextlib import nullcontext

    for name in ("TYRELL_QWEN_TOOLS", "TYRELL_PAIR_ARM", "TYRELL_PAIR_SECONDS", "TYRELL_DIAGNOSTIC_ABLATIONS"):
        monkeypatch.delenv(name, raising=False)
    for name in postgres_loader.PG_CONNECTION_ENV:
        monkeypatch.setenv(name, "fixture")
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-only")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://model.invalid/v1")
    monkeypatch.setenv("TYRELL_TIME_LIMIT", "120")
    monkeypatch.setenv("TYRELL_RUNTIME_STATS", str(local_run / "runtime.json"))
    monkeypatch.setattr(solution, "TOTAL_SECONDS", solution.training_seconds(120))
    task = local_run / "task.md"
    task.write_text("Target: delay. ID: event_id. Metric: MAE. Events are ordered by timestamp.")

    def snapshot(task_path, folder, *, on_core_ready, **kwargs):
        for csv_file in ("train.csv", "test.csv", "sample_submission.csv"):
            shutil.copyfile(task_path.parent / csv_file, folder / csv_file)
        on_core_ready()

    monkeypatch.setattr(postgres_loader, "prepare_postgres", snapshot)
    # Discard the setup fixture's CSV so that requests check publication by the actual entry point.
    (local_run / "submission.csv").unlink()
    responses = [response("run_experiment", ARGS), response(text="done")]
    if mode == "malformed_spec":
        monkeypatch.setattr(solution, "guess_task_spec", lambda *args: (SPEC, False))
        responses = [response(text="invalid JSON"), response("run_experiment", ARGS)]
    if mode == "api_error":
        responses = [RuntimeError("private-error"), RuntimeError("private-error")]

    def answer(**kwargs):
        assert (local_run / "submission.csv").is_file()
        result = responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    invoke = Mock(side_effect=answer)
    mock = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=invoke)))
    factory = Mock(side_effect=lambda **kwargs: nullcontext(mock))
    monkeypatch.setattr(openai, "OpenAI", factory)

    solution.run_agent(task)
    state = json.loads((local_run / "runtime.json").read_text(encoding="utf-8"))
    journal = json.loads((local_run / ".tyrell_experiments.json").read_text(encoding="utf-8"))
    assert state["phase"] == "complete"
    qwen = state["qwen_tool_run"]
    assert qwen["attempted_calls"] == invoke.call_count == 2
    requests = [call.kwargs for call in invoke.call_args_list]
    assert all(req["max_tokens"] == 1024 for req in requests)
    assert ("tools" not in requests[0]) == (mode == "malformed_spec")
    assert all([t["function"]["name"] for t in req["tools"]] == ["run_experiment"]
               for req in requests if "tools" in req)
    available = mode != "api_error"
    assert factory.call_args.kwargs["max_retries"] == 0
    assert qwen["completed_experiments"] == int(available)
    assert qwen["status"] == ("ok" if available else "api_error")
    assert any(entry["saved"] and entry["complete"] for entry in journal)
    if available:
        trial = qwen["trials"][0]
        assert journal[trial["journal_index"]]["score"] == trial["score"]
    solution.check_result(local_run, SPEC)


def test_task_identification_shares_cap_with_native_experiment(local_run, monkeypatch):
    state = solution.diagnostic_usage()
    # The target is ambiguous to the local parser. Only declared task candidates may be accepted.
    summary = {"possible_targets": ["delay", "other"], "possible_id_columns": ["event_id"]}
    guess = ({"target_column": None, "id_columns": [], "task_type": None, "metric": None,
              "probabilities": False}, False)
    bad = SPEC | {"target_column": "not_in_train"}
    invoke = Mock(side_effect=[response(text=json.dumps(bad)), response(text=json.dumps(SPEC))])
    mock = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=invoke)))
    deadline = time.monotonic() + 100
    spec = solution.resolve_spec(mock, "task", summary, guess, deadline, state)
    assert spec == SPEC and state["attempted_calls"] == state["responses"] == 2
    solution.run_llm_loop(mock, "", {}, spec, local_run, deadline, tool_state=state)
    assert invoke.call_count == 2 and not state["completed_experiments"]
    assert all("tools" not in call.kwargs and call.kwargs["max_tokens"] == 1024 for call in invoke.call_args_list)
