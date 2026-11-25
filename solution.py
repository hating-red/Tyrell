import time

PROCESS_STARTED = time.monotonic()

import json
import os
from pathlib import Path
import subprocess
import sys

from time_budget import pop_worker_deadline, run_until, shutdown_reserve, supervised_exit, time_limit, worker_command


CLI_DEADLINE = None
if __name__ == "__main__":
    CLI_DEADLINE = pop_worker_deadline()
    if CLI_DEADLINE is None:
        limit = time_limit(os.environ.get("TYRELL_TIME_LIMIT", 7200))
        # Leave a small allowance to reap a blocked worker and exit before the platform cap.
        CLI_DEADLINE = PROCESS_STARTED + limit - shutdown_reserve(limit)
        raise SystemExit(supervised_exit(worker_command(__file__, CLI_DEADLINE, sys.argv[1:]), CLI_DEADLINE))

from contextlib import ExitStack, contextmanager
import logging
import tempfile

# Ограничения применяются до импорта pandas/numpy и наследуются процессами моделей.
for _setting in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_setting] = "8"

from data_tools import guess_task_spec, inspect_data
from pipeline import check_result, write_fallback


def training_seconds(limit=7200):
    limit = time_limit(limit)
    return limit - min(60.0, limit * 0.05)


TOTAL_SECONDS = training_seconds(os.environ.get("TYRELL_TIME_LIMIT", 7200))
HARD_DEADLINE = [None]                # общий дедлайн: ни один эксперимент его не переживет
SAFETY_SHARE = 0.06                   # быстрый страховочный ответ
AUTO_SHARE = 0.25                     # базовый auto_train до LLM
FINAL_SHARE = 0.2                     # остаток на финальный auto_train с лучшими признаками
# One budget covers task identification, tool requests and failed transport attempts.
MAX_LLM_CALLS = int(os.environ.get("TYRELL_MAX_LLM_CALLS", 2))
LLM_USAGE = {"calls": 0, "prompt": 0, "completion": 0}
FINAL_MAX_SEEDS = 8                   # сидов на тип модели в финальном auto_train
ENSEMBLE_RESERVE = 300                # секунд на смесь экспериментов в конце


def log(message):
    print(message, file=sys.stderr, flush=True)


def run_pipeline_worker(operation, payload, task_dir, timeout, *, result_line=False):
    # Окружение наследуем, иначе часть библиотек не находит свои пакеты и HOME,
    # но ключи LLM в код агента не передаем
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith(("OPENAI_", "PG"))
           and k != "TYRELL_RUNTIME_STATS"}
    env |= {
        "PYTHONIOENCODING": "utf-8",
        "OMP_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "8",
    }
    env.setdefault("HOME", str(task_dir))
    deadline = time.monotonic() + timeout
    if HARD_DEADLINE[0] is not None:
        deadline = min(deadline, HARD_DEADLINE[0])
    try:
        # The interpreter locates sibling modules from the script path; data stays in the task directory.
        script = Path(__file__).resolve().with_name("pipeline.py")
        with tempfile.TemporaryFile() as request:
            request.write(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
            request.seek(0)
            result = run_until(
                [sys.executable, "-B", "-u", str(script), operation], deadline,
                cwd=task_dir, stdin=request, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                encoding="utf-8", errors="replace", env=env,
            )
        return {
            "stdout": (result.stdout.rstrip().rsplit("\n", 1)[-1] if result_line else result.stdout[-12000:]),
            "stderr": result.stderr[-12000:],
            "exit_code": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"error": "Превышено время выполнения кода."}


def checkpoint_history(task_dir):
    """Keep completed trials alongside the published CSV if a later model is interrupted."""
    source = task_dir / ".tyrell_experiments.json"
    if not source.is_file():
        return []
    history = json.loads(source.read_text(encoding="utf-8"))
    output = Path(os.environ.get("TYRELL_OUTPUT_DIR") or task_dir)
    if output != task_dir:
        visible = [{k: v for k, v in trial.items() if k != "preds_file"} for trial in history]
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output,
                                         prefix=".tyrell-history-", delete=False) as tmp:
            json.dump(visible, tmp, ensure_ascii=False)
        try:
            os.replace(tmp.name, output / source.name)
        finally:
            Path(tmp.name).unlink(missing_ok=True)
    return history


def experiment(task_dir, spec, budget, recipe="single", config=None, feature_code=None, *, deadline=None):
    """Эксперимент в отдельном процессе: падение или утечка памяти не задевают агента."""
    payload = {"task_spec": spec, "config": config or {}, "time_budget": budget,
               "recipe": recipe, "feature_code": feature_code}
    timeout = budget * 1.2 + min(90, TOTAL_SECONDS * 0.04)
    if HARD_DEADLINE[0] is not None:
        # Не дольше общего дедлайна: сабмит пишется атомарно, недоделанный шаг его не испортит
        timeout = min(timeout, HARD_DEADLINE[0] - time.monotonic())
    if deadline is not None:
        timeout = min(timeout, deadline - time.monotonic())
    if budget < 1 or timeout < 1:
        return None, {"error": "Общее время работы истекло."}
    result = run_pipeline_worker("experiment", payload, task_dir, timeout, result_line=True)
    checkpoint_history(task_dir)
    if result.get("exit_code") == 0:
        try:
            return json.loads(result["stdout"].strip().splitlines()[-1]), result
        except (ValueError, IndexError):
            pass
    return None, result


def ensemble(task_dir, spec, *, deadline=None):
    """Смесь лучших экспериментов в отдельном процессе; сабмит меняется только при приросте больше шума."""
    timeout = ENSEMBLE_RESERVE
    if HARD_DEADLINE[0] is not None:
        timeout = min(timeout, HARD_DEADLINE[0] - time.monotonic())
    if deadline is not None:
        timeout = min(timeout, deadline - time.monotonic())
    if timeout < (2 if deadline is not None else 30):
        return
    result = run_pipeline_worker("ensemble", {"task_spec": spec}, task_dir, timeout)
    checkpoint_history(task_dir)
    output = result.get("stdout", "").strip().splitlines()
    log(f"Ансамбль экспериментов: {output[-1] if result.get('exit_code') == 0 and output else str(result)[-1000:]}")


def track_usage(response):
    """Считаем токены: лимит LLM дневной, и по логу видно, сколько стоит прогон."""
    # Поля читаем осторожно: сервер может вернуть usage в другом виде, запрос это не должно ломать
    usage = getattr(response, "usage", None)
    LLM_USAGE["calls"] += 1
    LLM_USAGE["prompt"] += getattr(usage, "prompt_tokens", 0) or 0
    LLM_USAGE["completion"] += getattr(usage, "completion_tokens", 0) or 0
    log(f"LLM: запросов {LLM_USAGE['calls']}, токенов на входе {LLM_USAGE['prompt']}, "
        f"на выходе {LLM_USAGE['completion']}")


def diagnostic_usage():
    return {"status": "waiting", "attempted_calls": 0, "responses": 0, "tool_calls": 0,
            "completed_experiments": 0, "prompt_tokens": 0, "completion_tokens": 0, "trials": []}


@contextmanager
def quiet_api_logs():
    # The SDK and its HTTP transport may include the URL or headers in debug records.
    previous = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        yield
    finally:
        logging.disable(previous)


def diagnostic_tool():
    from paired_research import _PARAMETERS

    parameter_names = {key for limits in _PARAMETERS.values() for key in limits}
    config = {
        "model": {"type": "string", "enum": list(_PARAMETERS)},
        "params": {"type": "object", "properties": {key: {"type": "number"} for key in sorted(parameter_names)},
                   "additionalProperties": False},
        "cat_encoding": {"type": "string", "enum": ["native", "frequency"]},
        **{key: {"type": "boolean"} for key in ("auto_features", "entity_history", "categorical_ids")},
    }
    properties = {
        "config": {"type": "object", "properties": config, "minProperties": 1, "additionalProperties": False},
        "recipe": {"type": "string", "enum": ["single", "auto"]},
        "time_budget": {"type": "number", "minimum": 30, "maximum": 2400},
    }
    return {"type": "function", "function": {"name": "run_experiment",
            "description": "Train a candidate on the fixed validation folds and save it if accepted.",
            "parameters": {"type": "object", "properties": properties,
                           "required": list(properties), "additionalProperties": False}}}


def diagnostic_arguments(raw):
    from paired_research import _PARAMETERS

    args = json.loads(raw)
    if not isinstance(args, dict) or set(args) != {"recipe", "config", "time_budget"}:
        raise ValueError("invalid_arguments")
    cfg, recipe, budget = args["config"], args["recipe"], args["time_budget"]
    if (not isinstance(cfg, dict) or set(cfg) - {"model", "params", "cat_encoding", "auto_features",
                                                  "entity_history", "categorical_ids"}
            or recipe not in ("single", "auto") or type(budget) not in (int, float) or not 30 <= budget <= 2400):
        raise ValueError("invalid_configuration")
    model = cfg.get("model", "lightgbm")
    if not isinstance(model, str) or model not in _PARAMETERS or cfg.get("cat_encoding", "native") not in ("native", "frequency"):
        raise ValueError("invalid_backend_or_encoding")
    for flag in ("auto_features", "entity_history", "categorical_ids"):
        if flag in cfg and type(cfg[flag]) is not bool:
            raise ValueError("invalid_feature_flag")
    params = cfg.get("params", {})
    if not cfg or (set(cfg) == {"params"} and not params):
        raise ValueError("empty_hypothesis")
    if not isinstance(params, dict) or set(params) - _PARAMETERS[model].keys():
        raise ValueError("unknown_parameters")
    for key, value in params.items():
        kind, lower, upper = _PARAMETERS[model][key]
        if type(value) not in (int, float) or not lower <= value <= upper or (kind == "int" and type(value) is not int):
            raise ValueError("parameter_out_of_bounds")
    return args


def diagnostic_experiment(call, task_dir, spec, deadline, state):
    """Run a validated native tool call and corroborate its result with the local journal."""
    if call.function.name != "run_experiment":
        raise ValueError("unknown_tool")
    args = diagnostic_arguments(call.function.arguments)
    history_path = task_dir / ".tyrell_experiments.json"
    index = len(json.loads(history_path.read_text(encoding="utf-8")))
    slots = max(1, min(MAX_LLM_CALLS, 2) - state["attempted_calls"] + 1)
    budget = min(args["time_budget"], (deadline - time.monotonic()) / slots)
    trial = {"hypothesis": f"qwen-{len(state['trials']) + 1}", "tool": "run_experiment",
             "recipe": args["recipe"], "config": args["config"], "journal_index": None, "complete": False, "score": None}
    state["trials"].append(trial)
    if budget < 1:
        trial["error"] = "budget_exhausted"
        return args, trial
    result, _ = experiment(task_dir, spec, budget, recipe=args["recipe"], config=args["config"], deadline=deadline)
    history = json.loads(history_path.read_text(encoding="utf-8"))
    if result and len(history) == index + 1:
        written = history[index]
        if (written.get("recipe") == args["recipe"] and written.get("metric") == spec["metric"]
                and written.get("fold_signature") and written["fold_signature"] == result.get("fold_signature")
                and written.get("score") == result.get("score")):
            trial["journal_index"] = index
            trial["complete"] = bool(written.get("complete") and result.get("complete"))
            trial["score"] = written["score"]
            trial["saved"] = bool(written.get("saved"))
    if trial["complete"]:
        state["completed_experiments"] += 1
    else:
        trial["error"] = "experiment_incomplete"
    return args, trial


def ask_llm(client, messages, deadline, state, notify, *, tools=None):
    """A failed transport still consumes one of the two available attempts."""
    remaining = deadline - time.monotonic()
    if remaining < 12 or state["attempted_calls"] >= min(2, MAX_LLM_CALLS):
        return None
    state["attempted_calls"] += 1
    notify()
    options = {"tools": tools, "parallel_tool_calls": False} if tools else {}
    try:
        with quiet_api_logs():
            response = client.chat.completions.create(
                model=os.environ["OPENAI_MODEL"], messages=messages,
                max_tokens=1024, timeout=min(45, remaining - 8), **options)
        track_usage(response)
        state.update(responses=LLM_USAGE["calls"], prompt_tokens=LLM_USAGE["prompt"],
                     completion_tokens=LLM_USAGE["completion"])
        notify()
        return response
    except Exception as error:
        state["last_error"] = "api_error"
        log(f"LLM request failed: {type(error).__name__}")
        notify()
        return None


def compact_summary(summary):
    """Сводка для LLM без описаний колонок: они уже есть в task.md, а токены тратят дневной лимит."""
    out = {k: summary[k] for k in ("possible_targets", "possible_id_columns", "sample_columns") if k in summary}
    out["columns_format"] = "name, dtype, missing, unique"
    for part in ("train", "test"):
        info = summary.get(part, {})
        out[part] = {"rows": info.get("rows"), "columns": [
            [c["name"], c["dtype"], c["missing"], c["unique"]] for c in info.get("columns", [])]}
    return json.dumps(out, ensure_ascii=False)


def resolve_spec(client, task, summary, guess, deadline, state=None, notify=lambda: None):
    """Уверенной эвристике верим; иначе просим LLM уточнить task_spec и проверяем ответ."""
    spec, confident = guess
    if confident or client is None:
        return spec
    messages = [
        {"role": "system", "content": "Определи task_spec задачи. Ответь только JSON с ключами "
                                      "target_column, id_columns (список из одной колонки), task_type "
                                      "(regression, binary, multiclass), metric, probabilities."},
        {"role": "user", "content": task + "\n\nСводка данных:\n" + compact_summary(summary)
                                    + "\n\nПредположение по данным:\n" + json.dumps(spec, ensure_ascii=False)},
    ]
    if state is None:
        state = diagnostic_usage()
    # A usable local guess leaves at least one request for an experiment.
    attempts = 1 if spec.get("target_column") else 2
    for attempt in range(attempts):
        response = ask_llm(client, messages, deadline, state, notify)
        if response is None:
            continue
        try:
            text = response.choices[0].message.content or ""
            answer = json.loads(text[text.index("{"):text.rindex("}") + 1])
            candidate = spec | {k: answer[k] for k in spec if k in answer}
            from training import METRICS
            metric = METRICS.get(candidate.get("metric"))
            ids = candidate.get("id_columns")
            targets = summary.get("possible_targets", []) + [spec["target_column"]]
            identifiers = summary.get("possible_id_columns", []) + (spec.get("id_columns") or [])
            if (isinstance(candidate.get("target_column"), str) and candidate["target_column"] in targets
                    and isinstance(ids, list) and len(ids) == 1
                    and ids[0] in identifiers and metric and candidate.get("task_type") in metric.task_types
                    and isinstance(candidate.get("probabilities"), bool)):
                return candidate
        except Exception as error:
            log(f"Не удалось уточнить task_spec (попытка {attempt + 1}): {type(error).__name__}")
    return spec


def choose_variant(task_dir, spec, baseline, deadline):
    """Compare feature trials on fixed folds, then challenge the winning single model."""
    from training import compare_results

    if not baseline or not baseline.get("complete"):
        return None
    temporal = (baseline.get("split") or {}).get("type") == "time"
    applicable = baseline.get("feature_candidates") or {}
    feature_trials = [{flag: True} for flag in ("entity_history", "categorical_ids")
                      if temporal and applicable.get(flag)]
    calendar = {"auto_features": True}
    variants = (feature_trials + [calendar] if spec["task_type"] == "binary"
                else [calendar] + feature_trials)
    temporal_mae = temporal and spec["task_type"] == "regression" and spec.get("metric") == "mae"
    temporal_ap = temporal and spec["task_type"] == "binary" and spec.get("metric") == "pr_auc"
    if not temporal_mae:
        variants.append({"params": {"num_leaves": 31, "min_child_samples": 10}}
                        if temporal else {"target_encoding": True})

    def check(config, control):
        fit_seconds = max(1., float(control["fit_time"]))
        desired = max(8., 2.5 * fit_seconds + 5.)
        left = deadline - time.monotonic()
        if left < max(3., 1.15 * fit_seconds):
            return None
        report, raw = experiment(task_dir, spec, budget=min(left, desired), config=config, deadline=deadline)
        if report is None:
            log(f"Проверка {config} не завершилась: {str(raw)[-500:]}")
            return None
        signature = control.get("membership_signature")
        if (not report["complete"] or report["fold_signature"] != control["fold_signature"]
                or (signature and signature != report.get("membership_signature"))):
            log(f"Проверка {config}: нет сопоставимой полной валидации")
            return None
        comparison = compare_results(control, report)
        log(f"Проверка {config}: {report['score']} против {control['score']}, принят: {comparison['accept']}")
        return report if comparison["accept"] else None

    accepted = [(config, report) for config in variants if (report := check(config, baseline))]
    accepted.sort(key=lambda item: item[1]["score"], reverse=baseline["greater_is_better"])
    winner, incumbent = accepted[0] if accepted else ({}, baseline)
    if len(accepted) >= 2:
        combined = winner | accepted[1][0]
        if report := check(combined, incumbent):
            winner, incumbent = combined, report

    # Change one aspect of the best feature recipe and compare against that very model.
    if temporal_mae or temporal_ap:
        overrides = ({"min_child_samples": 100, "lambda_l2": 3.0} if temporal_mae
                     else {"metric": "average_precision"})
        candidate = winner | {"params": (winner.get("params") or {}) | overrides}
        if check(candidate, incumbent):
            return candidate
    return winner or None


def run_agent(task_path, *, deadline=None):
    limit = time_limit(os.environ.get("TYRELL_TIME_LIMIT", 7200))
    hard_deadline = deadline if deadline is not None else time.monotonic() + limit
    # Absolute across imports, PostgreSQL, every experiment and final CSV verification.
    deadline = hard_deadline - (limit - training_seconds(limit))
    path = os.environ.get("TYRELL_RUNTIME_STATS")
    if not path:
        return _run_agent(task_path, deadline)
    path = Path(path).resolve()
    state = {}

    def stats(**fields):
        state.update(fields)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".tyrell-runtime-", delete=False) as tmp:
            json.dump(state, tmp, ensure_ascii=False)
        try:
            os.replace(tmp.name, path)
        finally:
            Path(tmp.name).unlink(missing_ok=True)

    stats(phase="setup", solution_limit_seconds=limit,
          remaining_at_setup_seconds=round(max(0, hard_deadline - time.monotonic()), 3),
          remaining_for_research_seconds=round(max(0, deadline - time.monotonic()), 3))
    try:
        _run_agent(task_path, deadline, stats)
    except Exception as exc:
        stats(phase="failed", failed_phase=state["phase"], error_type=type(exc).__name__)
        raise
    stats(phase="complete")


def _run_agent(task_path, deadline, stats=None):
    from postgres_loader import PG_ENV, PG_CONNECTION_ENV, prepare_postgres

    # Keep the runner's task directory when task.md is a symlink.
    # A runner can link task.md into its working directory alongside the template.
    task_path = Path(task_path).absolute()
    task = task_path.read_text(encoding="utf-8-sig")
    HARD_DEADLINE[0] = deadline
    LLM_USAGE.update(calls=0, prompt=0, completion=0)
    use_pg = all(os.environ.get(key) for key in PG_CONNECTION_ENV)
    if not use_pg and any(os.environ.get(key) for key in PG_ENV):
        raise ValueError("Для PostgreSQL нужны PGHOST, PGPORT, PGUSER и PGDATABASE.")

    with ExitStack() as stack:
        task_dir = task_path.parent
        if use_pg:
            task_dir = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="tyrell-pg-")))
        previous_output = os.environ.get("TYRELL_OUTPUT_DIR")
        os.environ["TYRELL_OUTPUT_DIR"] = str(task_path.parent)
        if previous_output is None:
            stack.callback(os.environ.pop, "TYRELL_OUTPUT_DIR", None)
        else:
            stack.callback(os.environ.__setitem__, "TYRELL_OUTPUT_DIR", previous_output)

        client = None
        pair_arm = os.environ.get("TYRELL_PAIR_ARM")
        qwen_ready = MAX_LLM_CALLS > 0 and all(os.environ.get(name, "").strip() for name in
                                                ("OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL"))
        native = not pair_arm and (qwen_ready or os.environ.get("TYRELL_QWEN_TOOLS") == "1")
        tool_state = diagnostic_usage() if native else None

        def notify():
            if tool_state is not None and stats:
                stats(qwen_tool_run=tool_state)

        notify()
        if pair_arm != "automl" and qwen_ready:
            try:
                with quiet_api_logs():
                    from openai import DefaultHttpxClient, OpenAI
                    transport = stack.enter_context(DefaultHttpxClient(follow_redirects=False))
                    client = stack.enter_context(OpenAI(api_key=os.environ["OPENAI_API_KEY"],
                                                        base_url=os.environ["OPENAI_BASE_URL"],
                                                        max_retries=0, http_client=transport))
            except Exception:
                log("LLM client unavailable, using local models.")
                if tool_state is not None:
                    tool_state["status"] = "client_init_failed"
                    notify()
        else:
            log("Автономный режим, без запросов к языковой модели.")

        spec = None

        def save_early_submission():
            nonlocal spec
            try:
                guess = guess_task_spec(task_dir, task)
            except ValueError:
                if pair_arm or client is None:
                    raise
                guess = ({"target_column": None, "id_columns": [], "task_type": None,
                          "metric": None, "probabilities": False}, False)
            spec_client = None if pair_arm else client
            published = None
            if spec_client is not None and not guess[1] and guess[0].get("target_column"):
                # Publish the usable local interpretation before waiting for clarification.
                if not write_fallback(task_dir, guess[0]):
                    published = guess[0]
            summary = inspect_data(task_dir) if spec_client is not None and not guess[1] else {}
            candidate = resolve_spec(spec_client, task, summary, guess, deadline, tool_state, notify)
            if candidate != published:
                problems = write_fallback(task_dir, candidate)
                if problems:
                    raise ValueError("Некорректный страховочный ответ: " + ", ".join(problems))
            spec = candidate
            log(f"Страховочный submission.csv готов. task_spec: {spec}")

        if use_pg:
            pg_start = time.monotonic()
            if stats:
                stats(phase="postgres_connect")
            try:
                prepare_postgres(task_path, task_dir, deadline=deadline, log=log,
                                 on_core_ready=save_early_submission, on_stats=stats)
            except (RuntimeError, TimeoutError) as error:
                if spec is None:
                    raise
                log(f"Подготовка дополнительных данных прервана: {error}")
            finally:
                if stats:
                    stats(postgres_wall_seconds=round(time.monotonic() - pg_start, 3))
        else:
            save_early_submission()

        history_path = task_dir / ".tyrell_experiments.json"
        history_path.write_text("[]", encoding="utf-8")
        summary = {}
        if stats:
            stats(phase="research")
        research_start = time.monotonic()
        try:
            if tool_state is None:
                research(task_dir, spec, task, summary, client, deadline)
            else:
                research(task_dir, spec, task, summary, client, deadline, tool_state=tool_state, notify=notify)
        except Exception as error:
            if tool_state is not None:
                tool_state["status"] = "research_failed"
                notify()
                log(f"Цикл экспериментов прерван: {type(error).__name__}")
            else:
                log(f"Цикл экспериментов прерван: {type(error).__name__}: {error}")
        finally:
            if stats:
                stats(phase="validate_csv", research_seconds=round(time.monotonic() - research_start, 3))
        check_result(task_dir, spec)
        history = checkpoint_history(task_dir)
        saved = [h for h in history if h["saved"]]
        result = f"{spec['metric']}={saved[-1]['score']:.6g}" if saved else "страховочный прогноз"
        print(f"Готово: submission.csv проверен, {result}.", flush=True)


def research(task_dir, spec, task, summary, client, deadline, *, tool_state=None, notify=lambda: None):
    paired = bool(os.environ.get("TYRELL_PAIR_ARM"))
    scale = TOTAL_SECONDS
    if paired and "TYRELL_PAIR_SECONDS" in os.environ:
        scale = training_seconds(os.environ["TYRELL_PAIR_SECONDS"])
    compact = scale <= training_seconds(600)
    if client is not None and not paired and tool_state is None:
        tool_state = diagnostic_usage()
    search_seconds = 0.25 * scale
    final_reserve = min(30., 0.06 * scale) if compact else FINAL_SHARE * scale
    common_deadline = deadline - final_reserve - search_seconds
    initial_share = 0.12 if compact else SAFETY_SHARE
    safety, raw = experiment(task_dir, spec, budget=min(initial_share * scale, common_deadline - time.monotonic()),
                             deadline=common_deadline)
    if safety is None:
        log(f"Быстрая модель не завершилась: {str(raw)[-2000:]}")

    if not compact:
        budget = min(AUTO_SHARE * scale, common_deadline - time.monotonic())
        if budget > 15:
            report, raw = experiment(task_dir, spec, budget=budget, recipe="auto", deadline=common_deadline)
            log(f"Базовый auto_train: {report['score'] if report else str(raw)[-500:]}")

    estimated = max(3., float((safety or {}).get("fit_time") or 3.))
    # Preserve enough time for the two backends and an ensemble after single-model trials.
    auto_reserve = min(0.2 * scale, max(12., 3 * estimated))
    winner = choose_variant(task_dir, spec, safety, common_deadline - auto_reserve)
    if os.environ.get("TYRELL_DIAGNOSTIC_ABLATIONS") == "1":
        from diagnostic_ablation import run_ablations
        ablation_deadline = min(common_deadline - auto_reserve, time.monotonic() + 0.18 * scale)

        def ablation_experiment(*args, **kwargs):
            return experiment(*args, **kwargs, deadline=ablation_deadline)

        run_ablations(task_dir, spec, safety, ablation_deadline, ablation_experiment)

    budget = common_deadline - time.monotonic()
    config = {"max_seeds": 1} | (winner or {}) if compact else winner
    if not compact:
        budget = min(budget, AUTO_SHARE * scale)
    if config and budget > 15:
        report, raw = experiment(task_dir, spec, budget=budget, recipe="auto", config=config, deadline=common_deadline)
        log(f"auto_train с {config}: {report['score'] if report else str(raw)[-500:]}")

    if tool_state is not None:
        search_deadline = min(deadline - final_reserve, time.monotonic() + search_seconds)
        if client is None:
            if tool_state["status"] == "waiting":
                tool_state["status"] = "qwen_unavailable"
            notify()
        else:
            run_llm_loop(client, task, summary, spec, task_dir, search_deadline, tool_state=tool_state, notify=notify)
    else:
        from paired_research import run_paired_search
        search_deadline = min(deadline - final_reserve, time.monotonic() + search_seconds)
        run_paired_search(task_dir, spec, client, search_deadline, experiment)

    history = checkpoint_history(task_dir)
    if sum(bool(h.get("preds_file")) for h in history) >= 2:
        ensemble(task_dir, spec)
    remaining = deadline - time.monotonic()
    budget = (remaining - min(180, 0.04 * scale) - min(ENSEMBLE_RESERVE, 0.06 * scale)) / 1.1
    if not compact and budget > 300:
        saved = [h for h in history if h["saved"] and h["recipe"] != "ensemble"]
        best = saved[-1] if saved else None
        config = {"max_seeds": FINAL_MAX_SEEDS}
        if best:
            for key in ("cat_encoding", "params", "model", "entity_history", "categorical_ids"):
                if key in best["config"]:
                    config[key] = best["config"][key]
            config["auto_features"] = bool(best.get("auto_features"))
            config["target_encoding"] = best.get("target_encoding") or False
        report, raw = experiment(task_dir, spec, budget=budget, recipe="auto", config=config,
                                 feature_code=best and best.get("feature_code"))
        log(f"Финальный auto_train: {report['score'] if report else str(raw)[-500:]}")
        if report:
            ensemble(task_dir, spec)

    # Preserve the agreed search budgets in paired AutoML/Qwen comparisons.
    if not paired:
        search_remaining_budget(task_dir, spec, deadline)


def search_remaining_budget(task_dir, spec, deadline):
    """Spend the remaining wall time on independent batches on the original folds."""
    history = checkpoint_history(task_dir)
    controls = [h for h in history if h["complete"] and h["recipe"] in ("single", "auto")
                and not h.get("diagnostic_group") and not h.get("problems")]
    accepted = [h for h in controls if h["saved"]]
    if not accepted:
        return
    anchor = accepted[-1]
    # Reuse the accepted features. Each backend owns its parameter distribution.
    features = {k: anchor["config"][k] for k in ("cat_encoding", "categorical_ids", "entity_history")
                if k in anchor["config"]}
    features.update(auto_features=bool(anchor.get("auto_features")),
                    target_encoding=anchor.get("target_encoding") or False, refit_full=True)

    families = ["lightgbm", "xgboost", "catboost"]
    # Keep the number of outer-fold comparisons small even on a fast, small dataset.
    batch_window = max(12., (deadline - time.monotonic()) / 16)
    measured = {h["config"].get("model", "lightgbm"): h["fit_time"] for h in controls if h["recipe"] == "single"}
    typical = measured.get("lightgbm", 3.)
    overhead, blend_seconds = 3., 8.
    batches, pending = 0, 0
    while families:
        remaining = deadline - time.monotonic() - blend_seconds
        if remaining < max(8., 2 * overhead + 3.):
            break
        model = families.pop(0)
        started = time.monotonic()
        window = max(batch_window, 2.5 * measured.get(model, typical) + overhead)
        budget = min(window, remaining - overhead)
        cfg = features | {"model": model, "seed": 10007 + batches * 100003}
        report, error = experiment(task_dir, spec, budget=budget, config=cfg, recipe="bag",
                                   feature_code=anchor.get("feature_code"), deadline=deadline - blend_seconds)
        batches += 1
        if not report or not report["complete"]:
            reason = (error or {}).get("error") or (error or {}).get("stderr", "")[-300:]
            log(f"Дополнительный поиск: {model}, нет полного результата. {reason}")
            continue
        overhead = max(3., time.monotonic() - started - report["fit_time"] + 2.)
        families.append(model)
        pending += 1
        count = (report.get("members") or {}).get("count", 0)
        log(f"Дополнительный поиск: {model}, полных моделей {count}, "
            f"{spec['metric']}={report['score']:.6g}, сохранён={report.get('saved', False)}")
        if pending >= 3:
            started = time.monotonic()
            ensemble(task_dir, spec, deadline=min(deadline, started + 2 * blend_seconds))
            blend_seconds = min(ENSEMBLE_RESERVE, max(4., 1.5 * (time.monotonic() - started) + 2.))
            pending = 0

    if pending:
        ensemble(task_dir, spec, deadline=deadline)
    if batches:
        log(f"Дополнительный поиск завершён: серий {batches}, до рабочего дедлайна "
            f"{max(0., deadline - time.monotonic()):.1f} с.")


def run_llm_loop(client, task, summary, spec, task_dir, deadline, *, tool_state=None, notify=lambda: None):
    from paired_research import _PARAMETERS

    state = tool_state if tool_state is not None else diagnostic_usage()

    history = json.loads((task_dir / ".tyrell_experiments.json").read_text(encoding="utf-8"))
    previous = [{"recipe": h["recipe"], "model": h.get("config", {}).get("model"),
                 "auto_features": bool(h.get("auto_features")), "score": h["score"],
                 "fold_scores": h.get("fold_scores")} for h in history[-8:] if h.get("complete")]
    messages = [
        {"role": "system", "content": "Propose a testable ML hypothesis and call run_experiment with its configuration. "
                                     "Choose one supported model and its numeric parameters per request. "
                                     "Use the previous evaluation to propose a different experiment."},
        {"role": "user", "content": json.dumps({"task_type": spec["task_type"], "metric": spec["metric"],
                                    "validation": previous, "allowed_params": _PARAMETERS}, allow_nan=False)},
    ]
    state["status"] = "running"
    notify()
    while state["attempted_calls"] < min(2, MAX_LLM_CALLS) and deadline - time.monotonic() >= 12:
        response = ask_llm(client, messages, deadline, state, notify, tools=[diagnostic_tool()])
        if response is None:
            continue
        calls = response.choices[0].message.tool_calls if response.choices else None
        state["tool_calls"] += len(calls or ())
        if not calls:
            if state["completed_experiments"]:
                break
            state["last_error"] = "text_only"
            messages.append({"role": "user", "content": "Use the run_experiment function to test your hypothesis."})
            notify()
            continue
        try:
            if len(calls) != 1 or not isinstance(calls[0].id, str) or not calls[0].id:
                raise ValueError("invalid_tool_call")
            call = calls[0]
            args, trial = diagnostic_experiment(call, task_dir, spec, deadline, state)
        except (ValueError, TypeError, KeyError, AttributeError, OSError):
            state["last_error"] = "invalid_or_failed_tool_call"
            messages.append({"role": "user", "content": "Use exactly one supported run_experiment call and valid arguments."})
            notify()
            continue
        if trial["complete"]:
            state.pop("last_error", None)
        else:
            state["last_error"] = trial["error"]
        function = {"name": "run_experiment", "arguments": json.dumps(args, allow_nan=False)}
        assistant = {"role": "assistant", "content": None,
                     "tool_calls": [{"id": call.id, "type": "function", "function": function}]}
        # Echo provider-supplied tool-turn metadata only within this conversation.
        reasoning = getattr(response.choices[0].message, "reasoning_content", None)
        if isinstance(reasoning, str) and reasoning:
            assistant["reasoning_content"] = reasoning
        messages.extend([
            assistant,
            {"role": "tool", "tool_call_id": call.id, "content": json.dumps(trial, allow_nan=False)},
        ])
        notify()
    if state["completed_experiments"]:
        state["status"] = "partial" if state.get("last_error") else "ok"
    else:
        state["status"] = state.get("last_error", "no_completed_experiment")
    notify()


if __name__ == "__main__":
    run_agent(Path(sys.argv[1] if len(sys.argv) > 1 else "task.md"), deadline=CLI_DEADLINE)
