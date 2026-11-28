"""Competition entry point: python3 solution.py task.md."""
import time
PROCESS_STARTED = time.monotonic()

import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import traceback
from itertools import product

if __name__ == "__main__":
    sys.dont_write_bytecode = True
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from publication import publication_run, published_result, receipt_directory
from time_budget import limit_openmp, pop_worker_deadline, run_until, shutdown_reserve, supervised_exit, time_limit, worker_command

limit_openmp()


def log(message):
    print(message, file=sys.stderr, flush=True)


def log_phase(name, started, *, budget=None, reason, **details):
    payload = {"phase": name, "seconds": round(time.monotonic() - started, 3), "budget_seconds": budget,
               "reason": reason, **details}
    log("PHASE " + json.dumps(payload, ensure_ascii=False))


def trial_fits(estimate, deadline):
    remaining = deadline - time.monotonic()
    overhead = min(30., max(2., remaining * .05))
    return 1.3 * estimate + overhead <= remaining


def run_trial(directory, spec, deadline, budget, config=None, recipe="single", *, estimate=None):
    """Train in an isolated process. Keep the last complete CSV if a later trial fails."""
    started = time.monotonic()
    remaining = deadline - started
    overhead = min(30., max(2., remaining * .05))
    if not trial_fits(budget if estimate is None else estimate, deadline):
        return None
    budget = min(budget, (remaining - overhead) / 1.3)
    if budget < 1:
        return None
    request = {"task_spec": spec, "config": config or {}, "recipe": recipe, "time_budget": budget}
    from pipeline import output_directory
    request.update(output_dir=output_directory.get(), receipt_dir=receipt_directory.get())
    environment = {k: v for k, v in os.environ.items()
                   if not k.upper().startswith("OPENAI_") and k.upper() != "PGPASSWORD"}
    script = Path(__file__).resolve().with_name("pipeline.py")
    try:
        with tempfile.TemporaryFile() as stream:
            stream.write(json.dumps(request, ensure_ascii=False).encode("utf-8"))
            stream.seek(0)
            proc = run_until([sys.executable, "-B", "-u", str(script)], min(deadline, started + budget + overhead),
                             cwd=directory, env=environment, stdin=stream, stdout=subprocess.PIPE)
        if proc.returncode == 0:
            report = json.loads(proc.stdout.decode("utf-8").splitlines()[-1])
            if "skipped" in report:
                log(f"{recipe} {config or {}} skipped: {report['skipped']}")
                return None
            report["elapsed_seconds"] = time.monotonic() - started
            if recipe == "tune":
                return report
            log(f"{recipe} {config or {}}: {report['metric']}={report['score']:.6f}, "
                f"folds={report['n_folds_done']}, saved={report['saved']}")
            prefix = "AP_TRIAL_FOLDS" if report["metric"] == "pr_auc" else "TRIAL_FOLDS"
            log(f"{prefix} recipe={recipe} config={config or {}} "
                f"scores={report['fold_scores']} saved={report['saved']}")
            return report
    except Exception:
        log_error()
    return None


_CV_TOOL = "run_cv_experiment"
_CV_FLAGS = {"auto_features", "event_activity", "causal_history", "hourly_extended", "log_target", "mae_shift"}
_BACKENDS = ("lightgbm", "catboost", "xgboost")
_LEVEL1 = ("hgb", "linear")


def parse_model_object(content):
    """Read a JSON object following a thinking block or inside a fenced answer."""
    content = re.split(r"</think\s*>", content, flags=re.IGNORECASE)[-1]
    blocks = re.findall(r"\x60{3}(?:json)?\s*(.*?)\x60{3}", content, re.IGNORECASE | re.DOTALL)
    decoder = json.JSONDecoder()
    for text in [*blocks, content]:
        for match in re.finditer(r"\{", text):
            try:
                result, _ = decoder.raw_decode(text[match.start():])
            except json.JSONDecodeError:
                continue
            if isinstance(result, dict):
                return result
    raise ValueError("The model did not return a JSON object")


_LLM_ROLES = {
    "analyst": (25, 800), "feature_engineer": (35, 1500), "planner": (20, 400), "critic": (20, 600),
}
_LLM_INPUT_LIMIT = 12_000
_LLM_REQUEST_OVERHEAD = 128
_LLM_ROLE_GUIDANCE = {
    "feature_engineer": "Prioritize audited feature changes.",
    "planner": "Prioritize the next experiment within the remaining time.",
    "critic": "Use the existing evidence to avoid weak or redundant experiments.",
}


class TokenLedger:
    """A task budget partitioned into independently enforced role budgets."""
    def __init__(self, budget=40_000):
        self.budget = max(0, int(budget))
        self.limits = {role: self.budget * percent // 100 for role, (percent, _) in _LLM_ROLES.items()}
        self.usage = {role: {"in": 0, "out": 0, "calls": 0} for role in self.limits}

    def remaining(self, role):
        spent = {key: result["in"] + result["out"] for key, result in self.usage.items()}
        return self.budget - sum(spent.values()), self.limits[role] - spent[role]

    def allows(self, role, estimated_input, max_output):
        return min(self.remaining(role)) >= estimated_input + max_output

    def charge(self, role, prompt_tokens, completion_tokens):
        record = self.usage[role]
        record["in"] += prompt_tokens
        record["out"] += completion_tokens
        record["calls"] += 1
        task, remaining = self.remaining(role)
        log(f"LLM role={role} in={prompt_tokens} out={completion_tokens} left_task={task} left_role={remaining}")


def _llm_schema(kind, payload):
    if kind == "hypothesis":
        fields = {flag: {"type": "boolean"} for flag in sorted(_CV_FLAGS)}
        fields["model"] = {"type": "string", "enum": payload.get("spec", {}).get("available_backends", _BACKENDS)}
        required = ["model"]
    else:
        evidence = payload.get("schema", {})
        fields = {
            "target_column": {"type": "string", "enum": evidence.get("possible_targets", ())[:20]},
            "id_column": {"type": "string", "enum": evidence.get("possible_id_columns", ())[:20]},
            "metric": {"type": "string", "enum": payload.get("metrics", ())[:30]},
            "task_type": {"type": "string", "enum": ("binary", "multiclass", "regression")},
            "probabilities": {"type": "boolean"},
        }
        required = ["target_column", "id_column", "metric", "task_type"]
    return {"type": "object", "properties": fields, "required": required, "additionalProperties": False}


def _validate_llm_object(value, schema):
    if not isinstance(value, dict) or not set(schema["required"]) <= value.keys() or not value.keys() <= schema["properties"].keys():
        raise ValueError("Invalid LLM object fields")
    for key, item in value.items():
        field = schema["properties"][key]
        expected = bool if field["type"] == "boolean" else str
        if type(item) is not expected or ("enum" in field and item not in field["enum"]):
            raise ValueError("Invalid LLM object value")
    return value


def _llm_tools(schema):
    return [{"type": "function", "function": {"name": _CV_TOOL, "description": "Propose a CV experiment.",
                                               "parameters": schema}}]


def _llm_request_text(messages, tools):
    return json.dumps([messages, tools], ensure_ascii=False, separators=(",", ":"))


def _bounded_llm_messages(kind, prompt, payload, schema, role, *, max_bytes=None):
    """Whitelist a schema summary and bounded experiment metadata, never source rows."""
    evidence = payload.get("schema", {})
    columns = [str(value)[:100] for value in payload.get("features", ())[:60]]
    spec = {key: payload.get("spec", {})[key] for key in ("task_type", "metric", "available_backends")
            if key in payload.get("spec", {})}
    trials = []
    for trial in payload.get("trials", ())[-10:]:
        cfg = trial.get("config", {})
        row = {"config": {key: value for key, value in cfg.items() if key in _CV_FLAGS | {"model"}
                          and isinstance(value, (str, bool))}}
        if isinstance(trial.get("scores"), (list, tuple)):
            row["scores"] = [value for value in trial["scores"][:10] if isinstance(value, (int, float))
                             and math.isfinite(value)]
        trials.append(row)
    summary = {"features": columns, "feature_count": len(payload.get("features", ())), "spec": spec,
               "schema": {key: [str(v)[:100] for v in evidence.get(key, ())[:20]]
                          for key in ("possible_targets", "possible_id_columns", "sample_columns")},
               "metrics": list(payload.get("metrics", ()))[:30], "trials": trials}
    task = str(payload.get("task", ""))
    system = prompt[:1500] + " " + _LLM_ROLE_GUIDANCE.get(role, "")
    if kind == "spec":
        system += " Return one JSON object matching this schema: " + json.dumps(schema, ensure_ascii=False)
    else:
        system += " Return one JSON object or call the declared CV tool."
    text = json.dumps(summary, ensure_ascii=False, separators=(",", ":"))
    # Size task prose separately so trimming can never produce invalid JSON.
    remaining = max(0, _LLM_INPUT_LIMIT - len(system) - len(text) - 80)
    task = task[:remaining]
    tools = _llm_tools(schema) if kind == "hypothesis" else []
    while True:
        content = json.dumps({"task": task, **summary}, ensure_ascii=False, separators=(",", ":"))
        messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
        excess = len(system) + len(content) - _LLM_INPUT_LIMIT
        if max_bytes is not None:
            request_size = len(_llm_request_text(messages, tools).encode("utf-8"))
            excess = max(excess, request_size - max_bytes)
        if excess <= 0:
            return messages
        if not task:
            return None
        task = task[:max(0, len(task) - max(1, excess // 2))]


class ResearchModel:
    """Research client with a single metered transport in the trusted broker."""
    def __init__(self, mailbox=None):
        self.remaining = {"spec": 1, "hypothesis": 8}
        self.mailbox = mailbox
        self.mode = os.environ.get("TYRELL_LLM", "live").strip().lower()
        self.enabled = self.mode == "live" and (mailbox is not None or all(
            os.environ.get(k) for k in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL")))
        try:
            budget = int(os.environ.get("TYRELL_LLM_TOKENS", "40000"))
        except ValueError:
            budget = 40_000
        self.ledger = TokenLedger(budget)
        self.extra_supported = True
        self.last_tool = None
        self.last_reason = None

    def _disable(self, reason):
        self.enabled = False
        self.last_reason = reason
        log(f"LLM status=disabled reason={reason}")

    def report_usage(self):
        log("LLM_USAGE " + json.dumps({"mode": self.mode, "budget": self.ledger.budget,
                                       "roles": self.ledger.usage}, separators=(",", ":")))

    def ask(self, kind, prompt, payload, deadline):
        self.last_tool = None
        self.last_reason = None
        if not self.enabled or self.remaining[kind] <= 0:
            return None
        count = self.remaining[kind]
        self.remaining[kind] -= 1
        if self.mailbox is not None:
            request = self.mailbox / f"{kind}-{count}.request"
            reply = request.with_suffix(".reply")
            staged = request.with_suffix(".tmp")
            staged.write_text(json.dumps([kind, prompt, payload, deadline], default=str), encoding="utf-8")
            staged.replace(request)
            while time.monotonic() < deadline:
                if reply.exists():
                    result, self.enabled, self.last_tool, self.last_reason = json.loads(reply.read_text(encoding="utf-8"))
                    return result
                time.sleep(.1)
            return None
        role = "analyst" if kind == "spec" else ("planner", "feature_engineer", "critic")[(8 - count) % 3]
        schema = _llm_schema(kind, payload)
        max_bytes = self.ledger.limits[role] - _LLM_ROLES[role][1] - _LLM_REQUEST_OVERHEAD
        messages = _bounded_llm_messages(kind, prompt, payload, schema, role, max_bytes=max_bytes)
        if not messages:
            self.last_reason = "token_budget"
            log(f"LLM role={role} status=skipped reason=token_budget")
            return None
        return self.llm_call(role, messages, schema, deadline=deadline, using_tool=kind == "hypothesis")

    def llm_call(self, role, messages, schema, *, deadline, using_tool=False):
        self.last_tool = None
        self.last_reason = None
        if self.mode != "live" or not self.enabled:
            return None
        model = os.environ["OPENAI_MODEL"]
        key = hashlib.sha256(json.dumps([model, role, messages], ensure_ascii=False,
                                        sort_keys=True).encode("utf-8")).hexdigest()
        root = Path(os.environ.get("TYRELL_LLM_CACHE", str(Path.home() / ".tyrell" / "llm-cache")))
        path = root / (key + ".json")
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            result = _validate_llm_object(cached["result"], schema)
            self.last_tool = _CV_TOOL if using_tool and cached.get("used_tool") else None
            log(f"LLM role={role} status=cache_hit")
            return result
        except (OSError, ValueError, KeyError, TypeError):
            pass

        from openai import APIStatusError, OpenAI

        max_output = _LLM_ROLES[role][1]
        tools = _llm_tools(schema) if using_tool else []
        serialized = _llm_request_text(messages, tools)
        if sum(len(message["content"]) for message in messages) > _LLM_INPUT_LIMIT:
            self.last_reason = "input_limit"
            return None
        estimated_input = math.ceil(len(serialized) / 3.5) + 24
        reserved_input = max(estimated_input, len(serialized.encode("utf-8")) + _LLM_REQUEST_OVERHEAD)
        for attempt in range(2):
            if deadline - time.monotonic() <= 5:
                self.last_reason = "deadline"
                return None
            if not self.ledger.allows(role, reserved_input, max_output):
                self.last_reason = "token_budget"
                log(f"LLM role={role} status=skipped reason=token_budget")
                return None
            options = {}
            if self.extra_supported:
                options["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
            if tools:
                options.update(tools=tools, tool_choice={"type": "function", "function": {"name": _CV_TOOL}})
            try:
                with OpenAI(base_url=os.environ["OPENAI_BASE_URL"], api_key=os.environ["OPENAI_API_KEY"],
                            timeout=min(90, deadline-time.monotonic()), max_retries=0) as client:
                    response = client.chat.completions.create(
                        model=model, max_tokens=max_output, messages=messages, **options)
            except APIStatusError as error:
                status = error.status_code
                body = getattr(error, "body", None)
                detail = (str(error) + " " + str(body)).lower()
                quota_errors = ("insufficient_quota", "quota_exceeded", "rate_limit", "context_length_exceeded",
                                "token_limit", "maximum context length", "tokens per minute")
                if status == 429 or any(code in detail for code in quota_errors):
                    self.ledger.charge(role, reserved_input, 0)
                    self._disable("rate_or_token_limit")
                    return None
                if status == 400 and self.extra_supported and attempt == 0:
                    # A rejected request has no completion. Charge its reserved input
                    # before checking the budget again for the sole compatibility retry.
                    self.ledger.charge(role, reserved_input, 0)
                    self.extra_supported = False
                    continue
                self.ledger.charge(role, reserved_input, 0)
                self._disable(f"http_{status}")
                return None
            except Exception:
                self.ledger.charge(role, reserved_input, max_output)
                self._disable("sdk_error")
                return None

            usage = getattr(response, "usage", None)
            actual_in = getattr(usage, "prompt_tokens", None)
            actual_out = getattr(usage, "completion_tokens", None)
            self.ledger.charge(role, actual_in if type(actual_in) is int and actual_in >= 0 else reserved_input,
                               actual_out if type(actual_out) is int and actual_out >= 0 else max_output)
            if min(self.ledger.remaining(role)) < 0:
                self._disable("reported_usage_exceeded_budget")
                return None
            try:
                message = response.choices[0].message
                calls = getattr(message, "tool_calls", None) or []
                if calls:
                    if not using_tool or len(calls) != 1 or calls[0].function.name != _CV_TOOL:
                        raise ValueError("Unexpected LLM tool")
                    value = json.loads(calls[0].function.arguments)
                else:
                    value = json.loads(message.content)
                result = _validate_llm_object(value, schema)
            except (ValueError, IndexError, TypeError, AttributeError):
                self.last_reason = "invalid_json_or_schema"
                log(f"LLM role={role} status=fallback reason=invalid_json_or_schema")
                return None
            self.last_tool = _CV_TOOL if calls else None
            try:
                root.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=root, delete=False) as write:
                    staged = Path(write.name)
                    json.dump({"result": result, "used_tool": bool(calls)}, write, ensure_ascii=False)
                try:
                    staged.replace(path)
                finally:
                    staged.unlink(missing_ok=True)
            except OSError:
                log(f"LLM role={role} status=cache_unavailable")
            return result
        return None


def serve_research_model(directory, stop, model=None):
    model = model or ResearchModel()
    while not stop.wait(.1):
        for request in directory.glob("*.request"):
            try:
                args = json.loads(request.read_text(encoding="utf-8"))
                result = model.ask(*args)
                reply = request.with_suffix(".writing")
                reply.write_text(json.dumps([result, model.enabled, model.last_tool, model.last_reason]), encoding="utf-8")
                reply.replace(request.with_suffix(".reply"))
                request.unlink()
            except Exception:
                log_error()
                return


def redact(detail):
    for key in ("OPENAI_API_KEY", "PGPASSWORD"):
        if os.environ.get(key):
            detail = detail.replace(os.environ[key], "[redacted]")
    return detail


def log_error():
    log(redact(traceback.format_exc()))


def resolve_task(directory, text, model, deadline, early=None):
    from data_tools import guess_task_spec, inspect_data, load_data, positive_class_spec, target_constraints, PROBA_METRICS
    from training import METRICS
    guess = None
    try:
        guess, certain = guess_task_spec(directory, text)
        if early:
            early(guess)
        if certain or not model.enabled:
            return guess
    except ValueError:
        log_error()
    summary = inspect_data(directory)
    evidence = {k: summary[k] for k in ("possible_targets", "possible_id_columns", "sample_columns")}
    while model.enabled and model.remaining["spec"] and deadline - time.monotonic() > 5:
        candidate = model.ask("spec", "Identify target_column, id_column, metric, task_type and probabilities "
                              "from these materials. Use one of the listed metrics.",
                              {"task":text[:9000],"schema":evidence,"metrics":list(METRICS)}, deadline)
        if not candidate:
            continue
        target, key, metric = (candidate.get(k) for k in ("target_column","id_column","metric"))
        kind = candidate.get("task_type")
        if not all(isinstance(c,str) for c in (target,key,metric,kind)):
            continue
        if (target not in evidence["possible_targets"] or key not in evidence["possible_id_columns"]
                or metric not in METRICS or kind not in METRICS[metric].task_types):
            continue
        spec = {"target_column":target,"id_columns":[key],"metric":metric,"task_type":kind,
                "probabilities":metric in PROBA_METRICS or len(evidence["sample_columns"])>2}
        data = load_data(directory, spec)
        if kind == "regression" and spec["probabilities"]:
            continue
        if kind in ("binary", "multiclass") and (data["y_train"].nunique() > 2) != (kind == "multiclass"):
            continue
        return spec | positive_class_spec(text, data["y_train"], kind) | target_constraints(text,target,data["y_train"],kind)
    if guess:
        return guess
    raise ValueError("Could not identify a supported task from the schema and template")


def hypothesis(model, task, spec, columns, trials, deadline):
    config = model.ask("hypothesis", "Propose ONE new CV experiment. Output a compact object such as "
                       '{"model":"xgboost","auto_features":true}. '
                       "Choose a model from available_backends. Optional boolean flags: auto_features, "
                       "event_activity, causal_history, hourly_extended, log_target, mae_shift. The last two are for MAE only. "
                       "Use prior results and avoid any configuration already tried.",
                       {"task":task[:6000],"spec":spec,"features":columns[:60],"trials":trials[-10:]}, deadline)
    flags = _CV_FLAGS
    if (config and set(config) <= flags | {"model"} and config.get("model") in spec.get("available_backends", _BACKENDS)
            and all(type(config[k]) is bool for k in flags & config.keys())):
        if spec["metric"] == "mae" or not (config.get("log_target") or config.get("mae_shift")):
            return config
    return None


def run_agent(task_path, directory, deadline):
    from pipeline import output_directory, schema_fallback, check_result
    from postgres_loader import _local_sample
    output = output_directory.set(str(task_path.parent))
    try:
        task = task_path.read_text(encoding="utf-8-sig")
        template = _local_sample(task_path, task)
        if template:
            schema_fallback(template, task, task_path.parent / "submission.csv")
        spec = research(task_path, directory, deadline)
        check_result(directory, spec)
    except Exception:
        log_error()
        if not published_result(task_path.parent / "submission.csv"):
            raise
    finally:
        output_directory.reset(output)


_STAGE_B_FEATURES = ("interval_features", "event_calendar", "fold_preprocess")


def _loss_variants(spec, baseline, accepted_features=()):
    if not baseline or not spec.get("available_backends"):
        return []
    backend = baseline.get("backend", spec["available_backends"][0])
    candidates = baseline.get("feature_candidates", {})
    full = {"auto_features": True} | {key: True for key in ("hourly_extended", "event_activity", "causal_history")
                                     if candidates.get(key)}
    variants = [full]
    for key in _STAGE_B_FEATURES:
        if key in accepted_features:
            full = full | {key: True}
            variants.append(full)
    if spec.get("task_type") == "regression" or spec["metric"] == "mae":
        if spec["metric"] != "mae":
            variants.append(full | {"mae_objective": "l1"})
        variants.append(full | {"mae_objective": "huber"})
        if spec["metric"] != "rmsle":
            variants.append(full | {"log_target": True})
        if candidates.get("sqrt_target"):
            variants.append(full | {"sqrt_target": True})
        if spec["metric"] == "mae":
            variants.append({"mae_shift": True})
    if spec["metric"] == "pr_auc":
        regularized = full | {"regularized_ap": True}
        variants.extend([regularized, regularized | {"es_metric": "logloss"}])
        if candidates.get("causal_history"):
            variants.extend([full | {"history_since_positive": False},
                             *(full | {"history_smoothing": m} for m in (5., 30.))])
    return [{"model": backend} | variant for variant in variants]


def choose_variant(spec, baseline, *, include_losses=True):
    if spec.get("audit_required"):
        models = spec["available_backends"]
        if not baseline or not models:
            return []
        backend = baseline.get("backend", models[0])
        candidates = baseline.get("feature_candidates", {})
        variants = [{"model": name} for name in _LEVEL1 if name in models and name != backend]
        variants += [{"auto_features": True}, {"rare_min_freq": 5}]
        variants.extend({key: True} for key in ("interval_features", "event_calendar") if candidates.get(key))
        variants.append({"fold_preprocess": True})
        if spec["metric"] == "pr_auc":
            variants.append({"es_metric": "logloss"})
        if candidates.get("hourly_extended"):
            variants.extend([{"hourly_extended": True}, {"auto_features": True, "hourly_extended": True}])
        if candidates.get("event_activity"):
            variants.append({"auto_features": True, "event_activity": True})
        if candidates.get("causal_history"):
            variants.extend([
                {"auto_features": True, "causal_history": True},
                {"auto_features": True, "event_activity": True, "causal_history": True},
            ])
        if ("catboost" in models and (baseline.get("split") or {}).get("type") == "time"
                and not spec.get("label_delay_unit")):
            if backend != "catboost":
                variants.append({"model": "catboost"})
            variants.append({"model": "catboost", "catboost_ordered_ctr": True})
        variants = [{"model": backend} | variant for variant in variants]
        if include_losses:
            variants.extend(option for option in _loss_variants(spec, baseline) if option not in variants)
        return variants
    variants = []
    if baseline and (baseline.get("split") or {}).get("type") == "time":
        available = baseline.get("feature_candidates", {})
        if available.get("entity_history"):
            variants.append({"auto_features": True, "entity_history": True})
        if spec["metric"] == "mae":
            variants.append({"auto_features": True, "log_target": True})
        if available.get("ad_history"):
            variants.extend([
                {"auto_features": True, "ad_history": True},
                {"auto_features": True, "ad_history": True, "entity_history": True},
            ])
    variants.extend([{"auto_features": True}, {"target_encoding": True}])
    backend = (baseline or {}).get("config", {}).get("model")
    if spec["metric"] == "pr_auc" and backend and backend != "lightgbm":
        return [{"model": backend} | variant for variant in variants]
    return variants


_SEARCH_FEATURES = ("auto_features", "causal_history", "event_activity", "hourly_extended",
                    "log_target", "mae_shift", "history_window", "history_smoothing", "mae_objective", "recency_days",
                    "rare_min_freq", "catboost_ordered_ctr", "es_metric", "class_weight", "interval_features",
                    "event_calendar", "fold_preprocess", "sqrt_target", "regularized_ap", "history_since_positive")


def _trial_key(config):
    cleaned = {key: value for key, value in config.items()
               if value is not None and value != {} and (value is not False or key == "history_since_positive")}
    if cleaned.get("seed") == 42:
        cleaned.pop("seed")
    return json.dumps(cleaned, sort_keys=True)


_FINAL_RESERVE = 400.
_CHECKPOINT_INTERVAL = 900.
_MAX_STALE_TRIALS = 8


def _extended_search(directory, spec, deadline, winner, tested, estimates, retired, *, checkpoint=None):
    from math import sqrt
    from pipeline import experiment_history
    from training import REGULARIZED_PARAMS, compare_results

    def rank():
        rows = [row for row in experiment_history(directory) if row.get("complete") and row.get("recipe") == "single"
                and row.get("quality_flags_pass") is not False]
        return sorted(rows, key=lambda row: -row["score"] if row["greater_is_better"] else row["score"])

    def model_config(row, model):
        raw = row["config"] if row else {key: winner[key] for key in _SEARCH_FEATURES if key in winner}
        return {key: raw[key] for key in (*_SEARCH_FEATURES, "params", "num_boost_round", "early_stopping_rounds")
                if key in raw and (raw[key] or key == "history_since_positive")} | {"model": model}

    def best_configs():
        rows = rank()
        return {model: model_config(next((row for row in rows if row["config"].get("model") == model), None), model)
                for model in estimates if model in REGULARIZED_PARAMS and retired.get(model) != "next_trial_does_not_fit"}

    def queue():
        for model, config in best_configs().items():
            budget = min(180., max(18., 1.8 * estimates[model]))
            if not trial_fits(budget + estimates[model], deadline):
                continue
            began = time.monotonic()
            report = run_trial(directory, spec, deadline, budget, config, "tune", estimate=budget)
            params = (report or {}).get("tuned_params")
            diagnostics = (report or {}).get("tune_diagnostics", {})
            log_phase("optuna_prefix", began, budget=budget, model=model,
                      **({"n_trials": 0, "completed": 0, "prefix_max": None} | diagnostics),
                      reason="parameters_selected" if params else "no_complete_trial")
            if params:
                yield config | {"params": params}, 1.25
        configs = best_configs()
        for model, config in configs.items():
            rate = "eta" if model == "xgboost" else "learning_rate"
            params = dict(config.get("params", {}))
            params.pop("learning_rate" if model == "xgboost" else "eta", None)
            yield config | {"params": params | {rate: .02}, "num_boost_round": 8000,
                            "early_stopping_rounds": 250}, 1.8
        for round_idx in range(2):
            for model, config in configs.items():
                yield config | {"params": config.get("params", {}) | REGULARIZED_PARAMS[model][round_idx]}, 1.15
        top = []
        for row in rank():
            model = row["config"].get("model")
            if model not in configs:
                continue
            config = model_config(row, model)
            if config not in top:
                top.append(config)
            if len(top) == 3:
                break
        for seed, config in product((101, 263, 419, 587), top):
            yield config | {"seed": seed}, 1.

    incumbent = next(iter(rank()), None)
    attempts = completed = stale = 0
    reason = "queue_exhausted"
    last_checkpoint = time.monotonic()
    for config, multiplier in queue():
        if time.monotonic() >= deadline:
            reason = "fit_deadline"
            break
        model = config["model"]
        estimate = multiplier * estimates[model]
        if not trial_fits(estimate, deadline):
            retired[model] = "next_trial_does_not_fit"
            continue
        key = _trial_key(config)
        if key in tested:
            continue
        tested.add(key)
        attempts += 1
        stale += 1
        report = run_trial(directory, spec, deadline, min(420., max(24., 1.55 * estimate)), config,
                           estimate=estimate)
        if report and report["complete"]:
            completed += 1
            estimates[model] = max(2., report.get("elapsed_seconds", report["fit_time"]), .75 * estimates[model])
            if incumbent is None:
                incumbent, stale = report, 0
            elif incumbent.get("fold_signature") == report.get("fold_signature"):
                paired = compare_results(incumbent, report)
                noise = paired["diff_std"] / sqrt(len(paired["diffs"]))
                if paired["diff_mean"] > noise:
                    incumbent, stale = report, 0
        if checkpoint and time.monotonic() - last_checkpoint >= _CHECKPOINT_INTERVAL:
            checkpoint()
            last_checkpoint = time.monotonic()
        if stale >= _MAX_STALE_TRIALS:
            reason = "stagnation"
            break
    if reason == "queue_exhausted" and time.monotonic() >= deadline:
        reason = "fit_deadline"
    for model in estimates:
        retired.setdefault(model, reason)
    return {"reason": reason, "families": retired, "attempts": attempts, "completed": completed,
            "extended_attempts": attempts, "extended_completed": completed, "consecutive_without_gain": stale}


def search_remaining_budget(directory, spec, deadline, baseline, winner, tested, *, blend_estimate=90.,
                            checkpoint=None):
    """Run a finite candidate queue before the deadline reserved for final publication."""
    fit_deadline = deadline - max(_FINAL_RESERVE, 1.3 * blend_estimate + 30.)
    estimate = max(2., baseline.get("elapsed_seconds", baseline["fit_time"]))
    models = spec.get("available_backends", _BACKENDS)
    estimates = dict.fromkeys(models, estimate)
    return _extended_search(directory, spec, fit_deadline, winner, tested, estimates, {}, checkpoint=checkpoint)


def research(task_path, directory, deadline):
    from pipeline import experiment_history, write_fallback
    from postgres_loader import prepare_postgres
    from leak_audit import AuditBudget, audit_log, restricted_inputs
    from training import backend_available

    search_enabled = deadline - time.monotonic() > _FINAL_RESERVE + 45.
    fit_deadline = deadline - _FINAL_RESERVE if search_enabled else deadline
    task = task_path.read_text(encoding="utf-8-sig")
    mailbox = directory / "research"
    model = ResearchModel(mailbox if mailbox.is_dir() else None)
    spec = None
    using_postgres = bool(os.environ.get("PGDATABASE"))
    replay = AuditBudget()
    pg_started = time.monotonic()
    join_started = None

    def core_ready():
        nonlocal spec, join_started
        if using_postgres:
            log_phase("pg_load", pg_started, reason="core_tables_loaded")
        safety_started = time.monotonic()
        def publish(guess):
            nonlocal spec
            problems = write_fallback(directory, guess | {"audit_required": True})
            if problems:
                raise ValueError(problems)
            spec = guess
        resolved = resolve_task(directory, task, model, fit_deadline, early=publish)
        if resolved != spec:
            publish(resolved)
        import csv
        with (directory / "test.csv").open(encoding="utf-8-sig", newline="") as stream:
            fields = next(csv.reader(stream))
        blocked = restricted_inputs(task, fields)
        backend_status = {}
        backends = [backend for backend in (*_BACKENDS, *_LEVEL1)
                    if backend_available(backend, diagnostics=backend_status)]
        log("BACKENDS " + " ".join(f"{name}={status}" for name, status in backend_status.items()))
        spec = spec | {"audit_required": True, "excluded_features": list(blocked), "available_backends": backends}
        for name, why in blocked.items():
            audit_log("excluded", "task.md", feature=name, reason=why)
        log_phase("safety_prior", safety_started, reason="prior_published")
        log(f"Task: {spec}")
        join_started = time.monotonic()

    if using_postgres:
        try:
            prepare_postgres(task_path, directory, deadline=fit_deadline, on_core_ready=core_ready, log=log, replay=replay)
        finally:
            if join_started is not None:
                log_phase("join", join_started, budget=round(max(0., fit_deadline - join_started)*.4, 3),
                          reason="join_plan_finished", leak_audit_seconds=replay.elapsed)
    else:
        # Local qualifier datasets also use this entry point.
        from data_tools import _read_files
        folder = _read_files(task_path.parent)[0]
        for name in ("train.csv", "test.csv", "sample_submission.csv", "data_dictionary.csv"):
            if (folder / name).is_file():
                shutil.copyfile(folder / name, directory / name)
        core_ready()

    import csv
    with (directory / "test.csv").open(encoding="utf-8-sig", newline="") as stream:
        columns = next(csv.reader(stream))
    quick = min(300., max(15., (fit_deadline - time.monotonic()) * .12))
    safety_started = time.monotonic()
    baseline = None
    baseline_trial = {}
    failed_trial = None
    for backend in spec["available_backends"]:
        baseline_trial = {"model": backend}
        baseline = run_trial(directory, spec, fit_deadline, quick, baseline_trial)
        if baseline:
            baseline["backend"] = backend
            break
        failed_trial = {"config": baseline_trial, "status": "baseline failed"}
    log_phase("safety_model", safety_started, budget=quick * (2 if failed_trial else 1),
              reason="model_completed" if baseline else "no_completed_model")
    blend_estimate = min(90., quick)
    last_checkpoint = safety_started

    def checkpoint():
        nonlocal blend_estimate, last_checkpoint
        history = experiment_history(directory)
        if sum(bool(h.get("preds_file")) and h.get("quality_flags_pass") is not False
               for h in history if h.get("recipe") != "ensemble") < 2:
            return
        started = time.monotonic()
        blend = run_trial(directory, spec, fit_deadline, 120., recipe="ensemble", estimate=blend_estimate)
        last_checkpoint = time.monotonic()
        if blend:
            blend_estimate = max(2., blend["elapsed_seconds"])
        log_phase("ensemble", started, budget=120., reason="blend_evaluated" if blend else "no_result",
                  checkpoint=True, saved=bool(blend and blend["saved"]))

    variant_started = time.monotonic()
    variants = choose_variant(spec, baseline, include_losses=False)
    losses_scheduled = False
    certified_features = set()
    winner = {}
    trials = [failed_trial] if failed_trial else []
    tested = {_trial_key(baseline_trial)} if baseline else set()
    unusable = 0
    if baseline:
        trials.append({"config": baseline_trial, "scores":baseline["fold_scores"]})
    search_deadline = time.monotonic() + .45 * max(0, fit_deadline-time.monotonic())
    llm_started = None
    llm_reason = "search_deadline"
    variant_trials = llm_trials = 0
    while time.monotonic() < search_deadline:
        from_llm = False
        if not variants and not losses_scheduled:
            losses_scheduled = True
            variants = _loss_variants(spec, baseline, certified_features)
        if variants:
            config = variants.pop(0)
        else:
            if llm_started is None:
                log_phase("choose_variant", variant_started, budget=round(search_deadline-variant_started, 3),
                          reason="queue_exhausted", trials=variant_trials)
                llm_started = time.monotonic()
            if not model.enabled:
                llm_reason = "llm_unavailable"
                break
            if not model.remaining["hypothesis"]:
                llm_reason = "request_budget_exhausted"
                break
            if not trial_fits(90., search_deadline):
                llm_reason = "next_request_does_not_fit"
                break
            config = hypothesis(model, task, spec, columns, trials, search_deadline)
            from_llm = True
            if not config:
                log("LLM_HYPOTHESIS status=no_valid_config")
                if model.last_reason == "invalid_json_or_schema":
                    llm_reason = model.last_reason
                    break
                unusable += 1
                if unusable >= 2:
                    llm_reason = model.last_reason or "no_valid_tool_configuration"
                    break
                continue
        key = _trial_key(config)
        if key in tested:
            if from_llm:
                log("LLM_HYPOTHESIS status=duplicate")
            continue
        tested.add(key)
        estimate = max(2., baseline.get("elapsed_seconds", baseline["fit_time"]) if baseline else quick)
        if not trial_fits(estimate, search_deadline):
            if from_llm:
                log("LLM_HYPOTHESIS status=insufficient_time")
                llm_reason = "next_trial_does_not_fit"
            break
        trial_budget = min(quick, 90.) if config.get("model") in _LEVEL1 else quick
        report = run_trial(directory, spec, search_deadline, trial_budget, config, estimate=estimate)
        if from_llm:
            llm_trials += 1
            status = ("completed" if report and report["complete"] else "incomplete")
            log(f"LLM_HYPOTHESIS status={status} config={config}")
            if model.last_tool and report:
                log(f"LLM_TOOL_EXECUTED name={model.last_tool} status={status}")
            unusable = 0 if status == "completed" else unusable + 1
        else:
            variant_trials += 1
        if report:
            if report.get("quality_flags_pass") is True:
                certified_features.update(key for key in _STAGE_B_FEATURES if config.get(key))
            trials.append({"config":config,"scores":report["fold_scores"]})
            if report["saved"]:
                winner = config
        else:
            trials.append({"config":config, "status":"skipped or failed"})
        if search_enabled and time.monotonic() - last_checkpoint >= _CHECKPOINT_INTERVAL:
            checkpoint()
        if unusable >= 2:
            llm_reason = "consecutive_incomplete_trials"
            break
    if llm_started is None:
        log_phase("choose_variant", variant_started, budget=round(search_deadline-variant_started, 3),
                  reason="search_deadline", trials=variant_trials)
        llm_started = time.monotonic()
    log_phase("llm_cycle", llm_started, budget=round(max(0., search_deadline-llm_started), 3),
              reason=llm_reason, trials=llm_trials)

    # Bound expensive searches so that completed candidates are published during the run.
    candidates = [{k:v for k,v in winner.items() if k not in ("model",)}, {"auto_features":True}]
    for index, candidate in enumerate(candidates):
        phase = "baseline_auto_train" if index == 0 else "final_auto_train"
        started = time.monotonic()
        if candidate in candidates[:index]:
            log_phase(phase, started, budget=0., reason="duplicate_configuration")
            continue
        remaining = fit_deadline-time.monotonic()
        if remaining < max(15.,1.3*(baseline["fit_time"] if baseline else 5.)):
            log_phase(phase, started, budget=0., reason="next_trial_does_not_fit")
            continue
        auto_budget = min(420., .3 * remaining, max(0., (remaining - 32.) / 1.3))
        report = run_trial(directory, spec, fit_deadline, auto_budget,
                           candidate | {"max_seeds": 5 if search_enabled else 3}, "auto")
        log_phase(phase, started, budget=round(auto_budget, 3),
                  reason=(report.get("auto_stop") or {}).get("reason", "no_completed_model") if report else "no_completed_model",
                  queue=(report or {}).get("auto_stop"))
        if report and report["saved"]:
            winner = candidate
    if search_enabled and baseline and baseline["complete"]:
        checkpoint()
        started = time.monotonic()
        search = search_remaining_budget(directory, spec, deadline, baseline, winner, tested,
                                         blend_estimate=blend_estimate, checkpoint=checkpoint)
        log_phase("search_remaining_budget", started, budget=round(max(0., deadline-started-_FINAL_RESERVE), 3),
                  final_reserve_seconds=max(_FINAL_RESERVE, 1.3 * blend_estimate + 30.), **search)
    else:
        log_phase("search_remaining_budget", time.monotonic(), budget=0., reason="not_enabled_for_task_or_budget")
    history = experiment_history(directory)
    started = time.monotonic()
    blend_budget = max(0., deadline-started)
    blend = None
    if not any(h.get("preds_file") and h.get("quality_flags_pass") is not False
               for h in history if h.get("recipe") != "ensemble"):
        blend_reason = "insufficient_completed_members"
    elif not trial_fits(blend_estimate, deadline):
        blend_reason = "next_trial_does_not_fit"
    else:
        blend = run_trial(directory, spec, deadline, blend_budget, {"refit": True},
                          recipe="ensemble", estimate=blend_estimate)
        blend_reason = "blend_evaluated" if blend else "no_result"
    log_phase("ensemble", started, budget=round(blend_budget, 3),
              reason=blend_reason, checkpoint=False)
    return spec


def main():
    deadline = pop_worker_deadline()
    task = Path(sys.argv[1] if len(sys.argv) > 1 else "task.md").absolute()
    if deadline is not None:
        receipt_directory.set(sys.argv[3])
        run_agent(task, Path(sys.argv[2]), deadline)
        return 0
    limit = time_limit(os.environ.get("TYRELL_TIME_LIMIT", 7200))
    shutdown = shutdown_reserve(limit)
    work_seconds = min(6500. - shutdown, limit - min(420., limit * 420 / 7200))
    deadline = PROCESS_STARTED + work_seconds
    with publication_run() as proofs, tempfile.TemporaryDirectory(prefix="tyrell-data-") as directory:
        cmd = worker_command(Path(__file__).resolve(), deadline, [str(task), directory, proofs])
        stop = threading.Event()
        llm = ResearchModel()
        broker = None
        if llm.enabled:
            mailbox = Path(directory) / "research"
            mailbox.mkdir()
            broker = threading.Thread(target=serve_research_model, args=(mailbox, stop, llm), daemon=True)
            broker.start()
        env = {k:v for k,v in os.environ.items() if not k.upper().startswith("OPENAI_")}
        if env.get("TYRELL_OOF_DIR"):
            archive = Path(env["TYRELL_OOF_DIR"]).resolve()
            try:
                archive.mkdir(parents=True, exist_ok=True)
                env["TYRELL_OOF_RUN_DIR"] = tempfile.mkdtemp(prefix=task.stem + "-", dir=archive)
                log(f"OOF_ARCHIVE {env['TYRELL_OOF_RUN_DIR']}")
            except OSError as error:
                log(f"OOF_ARCHIVE unavailable: {error}")
        try:
            code = supervised_exit(cmd, deadline + shutdown, env=env)
        finally:
            stop.set()
            if broker:
                broker.join(timeout=1.)
            llm.report_usage()
        output = task.parent / "submission.csv"
        receipt = published_result(output)
        from pipeline import experiment_history, schema_fallback, verify_template_output
        from postgres_loader import _local_sample
        task_text = task.read_text(encoding="utf-8-sig")
        sample = _local_sample(task, task_text) or Path(directory) / "sample_submission.csv"
        if sample.is_file():
            if not receipt:
                schema_fallback(sample, task_text, output)
                receipt = published_result(output)
            verify_template_output(sample, task_text, output)
        if not receipt:
            log("No validated predictions published by this run")
            return code or 1
        if receipt["state"] == "MODEL_SUCCESS":
            saved = [item for item in experiment_history(directory) if item["saved"]]
            if saved:
                winner = saved[-1]
                if winner["metric"] == "pr_auc":
                    log(f"AP_WINNER recipe={winner['recipe']} config={winner['config']}")
                for score, baseline in zip(winner["fold_scores"], winner.get("fold_baselines", [])):
                    better = score > baseline["score"] if winner["greater_is_better"] else score < baseline["score"]
                    constant = baseline["constant"]
                    constant = f"{constant:.6f}" if isinstance(constant, (int, float)) else json.dumps(constant, separators=(",", ":"))
                    log(f"CV_AUDIT metric={winner['metric']} fold={baseline['fold']} model={score:.6f} "
                        f"baseline={baseline['score']:.6f} constant={constant} "
                        f"train_rows={baseline['train_rows']} validation_rows={baseline['validation_rows']} "
                        f"model_beats_baseline={better}")
        print(f"SUBMISSION_STATE={receipt['state']} score={receipt.get('score')}", flush=True)
        log_phase("total", PROCESS_STARTED, budget=round(work_seconds+shutdown, 3),
                  reason=receipt["state"], platform_limit=limit)
        return 0


if __name__ == "__main__":
    sys.exit(main())
