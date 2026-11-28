import json
import time
from types import SimpleNamespace

import httpx
import openai
import pytest
import pandas as pd

from publication import MODEL_SUCCESS, publication_run, published_result
from solution import ResearchModel, TokenLedger, _bounded_llm_messages, _llm_schema, resolve_task, run_agent


def _response(content, *, tokens=(75, 24)):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                           usage=SimpleNamespace(prompt_tokens=tokens[0], completion_tokens=tokens[1]))


@pytest.fixture
def transport(monkeypatch, tmp_path):
    calls, items = [], []

    def create(**kwargs):
        calls.append(kwargs)
        item = items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    class Client:
        def __init__(self, **kwargs):
            assert kwargs["max_retries"] == 0
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

    monkeypatch.setenv("TYRELL_LLM", "live")
    monkeypatch.setenv("TYRELL_LLM_CACHE", str(tmp_path))
    monkeypatch.setenv("OPENAI_BASE_URL", "http://offline.invalid")
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test")
    monkeypatch.setenv("OPENAI_MODEL", "offline-model")
    monkeypatch.setattr(openai, "OpenAI", Client)
    return calls, items


def _call(model, prompt="one trial"):
    return model.ask("hypothesis", prompt, {}, time.monotonic() + 20)


def test_role_accounting_never_borrows_from_other_roles():
    ledger = TokenLedger(4000)
    assert ledger.limits == {"analyst": 1000, "feature_engineer": 1400, "planner": 800, "critic": 800}
    ledger.charge("analyst", 700, 120)
    assert ledger.remaining("analyst") == (3180, 180)
    assert not ledger.allows("analyst", 21, 160)
    assert ledger.allows("feature_engineer", 21, 160)


def test_ledger_counts_provider_usage_and_cache_is_shared_across_restarts(transport):
    calls, items = transport
    items.append(_response('{"model":"lightgbm"}'))
    first = ResearchModel()
    assert _call(first) == {"model": "lightgbm"}
    assert calls[0]["max_tokens"] == 400
    assert first.ledger.usage["planner"] == {"in": 75, "out": 24, "calls": 1}
    second = ResearchModel()
    assert _call(second) == {"model": "lightgbm"}
    assert len(calls) == 1 and second.ledger.usage["planner"]["calls"] == 0


def test_low_role_budget_prevents_transport(monkeypatch, transport):
    calls, _ = transport
    monkeypatch.setenv("TYRELL_LLM_TOKENS", "1500")
    model = ResearchModel()
    assert _call(model) is None
    assert model.last_reason == "token_budget"
    assert not calls


@pytest.mark.parametrize("mode", ["mock", "off"])
def test_offline_modes_never_invoke_transport_even_with_credentials(monkeypatch, transport, mode):
    calls, _ = transport
    monkeypatch.setenv("TYRELL_LLM", mode)
    model = ResearchModel()
    assert _call(model) is None and not calls
    assert all(not stats["calls"] for stats in model.ledger.usage.values())


def test_429_disables_later_attempts_without_network(transport, capsys):
    calls, items = transport
    request = httpx.Request("POST", "http://offline.invalid")
    items.append(openai.RateLimitError("quota exceeded", response=httpx.Response(429, request=request), body=None))
    model = ResearchModel()
    assert _call(model) is None
    assert not model.enabled and model.last_reason == "rate_or_token_limit"
    assert _call(model, "new experiment") is None and len(calls) == 1
    assert "LLM status=disabled reason=rate_or_token_limit" in capsys.readouterr().err


def test_rate_limit_still_produces_a_valid_submission(tmp_path, monkeypatch, transport, capsys):
    calls, items = transport
    request = httpx.Request("POST", "http://offline.invalid")
    items.append(openai.RateLimitError("quota exceeded", response=httpx.Response(429, request=request), body=None))
    monkeypatch.delenv("PGDATABASE", raising=False)
    frame = pd.DataFrame({"id": range(128), "signal": [i % 11 for i in range(128)]})
    frame.iloc[:112].assign(outcome=2.4 * frame["signal"].iloc[:112]).to_csv(tmp_path / "train.csv", index=False)
    frame.iloc[112:].to_csv(tmp_path / "test.csv", index=False)
    frame.iloc[112:][["id"]].assign(outcome=0).to_csv(tmp_path / "sample_submission.csv", index=False)
    task = tmp_path / "task.md"
    task.write_text("Predict the numeric target outcome for each id.", encoding="utf-8")
    worker = tmp_path / "worker"
    worker.mkdir()
    with publication_run():
        run_agent(task, worker, time.monotonic() + 36)
        assert published_result(tmp_path / "submission.csv")["state"] == MODEL_SUCCESS
    result = pd.read_csv(tmp_path / "submission.csv")
    assert result["id"].tolist() == list(range(112, 128))
    assert result["outcome"].notna().all()
    assert len(calls) == 1
    assert "LLM status=disabled reason=rate_or_token_limit" in capsys.readouterr().err


@pytest.mark.parametrize("content", ["broken", '{"model":"INVALID"}', '{"model":"hgb","extra":1}',
                                      'prefix {"model":"lightgbm"}'])
def test_invalid_json_or_schema_falls_back_without_retry(transport, content):
    calls, items = transport
    items.append(_response(content))
    model = ResearchModel()
    assert _call(model) is None
    assert model.last_reason == "invalid_json_or_schema"
    assert len(calls) == 1 and model.enabled


def test_invalid_analyst_output_uses_the_deterministic_spec(monkeypatch, tmp_path, transport):
    import data_tools

    calls, items = transport
    items.append(_response("invalid"))
    guess = {"target_column": "response", "id_columns": ["id"], "task_type": "regression", "metric": "mae"}
    monkeypatch.setattr(data_tools, "guess_task_spec", lambda *args: (guess, False))
    monkeypatch.setattr(data_tools, "inspect_data", lambda *_: {
        "possible_targets": ["response"], "possible_id_columns": ["id"], "sample_columns": ["id", "response"]})
    assert resolve_task(tmp_path, "task", ResearchModel(), time.monotonic() + 20) == guess
    assert len(calls) == 1 and calls[0]["max_tokens"] == 800


def test_bounded_context_whitelists_metadata_and_drops_raw_rows():
    payload = {"task": "описание" * 4000, "features": [f"column_{i}" for i in range(100)],
               "trials": [{"config": {"model": "hgb"}, "scores": [.3], "data": ["secret row"]}] * 25,
               "raw_rows": ["secret row"], "spec": {"task_type": "binary", "rows": ["secret row"]}}
    schema = _llm_schema("hypothesis", payload)
    messages = _bounded_llm_messages("hypothesis", "choose configuration", payload, schema, "planner")
    assert sum(len(message["content"]) for message in messages) <= 12000
    context = json.loads(messages[1]["content"])
    assert len(context["features"]) == 60 and len(context["trials"]) == 10
    assert "secret row" not in messages[1]["content"]


def test_long_unicode_task_still_uses_a_bounded_request(transport):
    calls, items = transport
    items.append(_response('{"model":"lightgbm"}'))
    model = ResearchModel()
    result = model.ask("hypothesis", "choose a model", {"task": "описание " * 2000}, time.monotonic() + 20)
    assert result == {"model": "lightgbm"}
    context = json.loads(calls[0]["messages"][1]["content"])
    assert 0 < len(context["task"]) < 12_000
    assert sum(d["in"] + d["out"] for d in model.ledger.usage.values()) <= model.ledger.budget
