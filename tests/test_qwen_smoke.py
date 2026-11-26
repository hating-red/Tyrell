import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest


path = Path(__file__).resolve().parents[1] / "runs" / "real-public" / "qwen_check.py"
spec = importlib.util.spec_from_file_location("tyrell_qwen_check", path)
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)

# Fictional values passed only to a local transport.
CONFIG = {"OPENAI_MODEL": "example/tool-alias", "OPENAI_BASE_URL": "https://gateway.invalid/custom/v2/",
          "OPENAI_API_KEY": "mock-access-value"}


def reply(tool=True):
    message = {"role": "assistant", "content": "private provider body"}
    if tool:
        message["tool_calls"] = [{"id": "call_x", "type": "function",
                                  "function": {"name": "smoke_ack", "arguments": "{}"}}]
    return {"id": "dummy", "object": "chat.completion", "created": 0, "model": "server-model",
            "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if tool else "stop"}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16}}


def test_local_check_does_not_initialize_client():
    with patch.object(check, "_request", side_effect=AssertionError("network path")):
        assert check.run("--check", CONFIG) == {"status": "configuration_valid", "api_requests": 0}


@pytest.mark.parametrize("url", ["https://mock-access-value@gateway.invalid", "file:///local",
                                 "https://gateway.invalid/v1?token=mock-access-value",
                                 "http://gateway.invalid/v1", "https://gateway.invalid:99999",
                                 "https://gateway.invalid\\@another.invalid", "https://invalid host"])
def test_malformed_or_unprotected_endpoint_fails_before_request(url):
    with patch.object(check, "_request", side_effect=AssertionError("network path")):
        assert check.run("--smoke", CONFIG | {"OPENAI_BASE_URL": url}) == {
            "status": "invalid_configuration", "variables": ["OPENAI_BASE_URL"]}


def test_configured_tunnel_is_allowed_without_a_request():
    assert check.run("--check", CONFIG | {"OPENAI_BASE_URL": "http://[::1]:13579/api"})["api_requests"] == 0


def test_one_bounded_request_preserves_config_and_reports_only_usage():
    requests = []

    def handle(request):
        requests.append(request)
        assert request.method == "POST"
        assert str(request.url) == CONFIG["OPENAI_BASE_URL"] + "chat/completions"
        assert request.headers["Authorization"] == "Bearer " + CONFIG["OPENAI_API_KEY"]
        body = json.loads(request.content)
        assert body["model"] == CONFIG["OPENAI_MODEL"]
        assert body["tools"][0]["function"]["name"] == "smoke_ack" and 0 < body["max_tokens"] <= 128
        assert request.extensions["timeout"]["read"] <= 20
        return httpx.Response(200, json=reply())

    report = check.run("--smoke", CONFIG, transport=httpx.MockTransport(handle))
    assert len(requests) == 1
    assert report["elapsed_seconds"] >= 0
    assert (report["status"], report["prompt_tokens"], report["completion_tokens"], report["total_tokens"]) == (
        "smoke_ok", 11, 5, 16)


@pytest.mark.parametrize("code", [307, 401, 429, 500])
def test_no_redirect_or_retry_and_sanitized_error(code, caplog, capsys):
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(code, headers={"Location": "https://redirect.invalid"},
                              json={"error": {"message": "private provider body " + CONFIG["OPENAI_API_KEY"]}})

    with caplog.at_level("DEBUG"):
        report = check.run("--smoke", CONFIG, transport=httpx.MockTransport(handle))
    assert len(requests) == 1
    assert report["status"] == "api_rejected_request" and report["http_status"] == code
    captured = capsys.readouterr()
    output = str(report) + caplog.text + captured.out + captured.err
    assert all(value not in output for value in [*CONFIG.values(), "private provider body", "redirect.invalid"])


def test_plain_assistant_response_is_not_a_tool_call():
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json=reply(tool=False)))
    assert check.run("--smoke", CONFIG, transport=transport)["status"] == "expected_tool_call_missing"


@pytest.mark.parametrize("verb,op,error,phase,kind", [
    (b"POST", "connect_tcp", httpx.ConnectTimeout, "dns_or_tcp", "connect"),
    (b"POST", "start_tls", httpx.ConnectTimeout, "tls", "connect"),
    (b"CONNECT", "receive_response_headers", httpx.ReadTimeout, "proxy_connect_response_headers", "read"),
    (b"POST", "receive_response_headers", httpx.ReadTimeout, "response_headers", "read"),
    (b"POST", "send_request_body", httpx.WriteTimeout, "send_body", "write"),
])
def test_timeout_trace_distinguishes_proxy_from_post(verb, op, error, phase, kind):
    from types import SimpleNamespace

    def handle(request):
        trace = request.extensions["trace"]
        if op not in ("connect_tcp", "start_tls"):
            trace("http11.send_request_headers.started", {"request": SimpleNamespace(method=verb)})
            if op == "receive_response_headers":
                trace("http11.send_request_body.started", {})
                trace("http11.send_request_body.complete", {})
        trace("connection." + op + ".started", {"headers": CONFIG["OPENAI_API_KEY"]})
        trace("http11.response_closed.started", {})
        raise error("private_provider_timeout", request=request)

    report = check.run("--smoke", CONFIG, transport=httpx.MockTransport(handle))
    assert report["status"] == "timeout" and report["phase"] == phase and report["timeout_kind"] == kind
    assert report["post_body_sent"] == (verb == b"POST" and op == "receive_response_headers")
    assert report["elapsed_seconds"] >= report["phase_seconds"] >= 0
    assert all(x not in str(report) for x in [*CONFIG.values(), "private_provider_timeout"])
