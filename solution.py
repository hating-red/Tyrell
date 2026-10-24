import json
import os
from pathlib import Path
import subprocess
import sys
import time

from openai import OpenAI

from data_tools import guess_task_spec, inspect_data
from pipeline import check_result

# Лимит платформы с запасом на завершение; TYRELL_TIME_LIMIT - только для локальных прогонов
# Запас 10 минут: платформа считает длительность вместе с запуском контейнера и данными
TOTAL_SECONDS = float(os.environ.get("TYRELL_TIME_LIMIT", 3 * 60 * 60)) - 600
HARD_DEADLINE = [None]                # общий дедлайн: ни один эксперимент его не переживет
SAFETY_SHARE = 0.06                   # быстрый страховочный ответ
AUTO_SHARE = 0.25                     # базовый auto_train до LLM
FINAL_SHARE = 0.2                     # остаток на финальный auto_train с лучшими признаками
LLM_ATTEMPTS = 4                      # попыток на один запрос к LLM
# Лимит запросов к LLM за прогон; TYRELL_MAX_LLM_CALLS - чтобы локальные тесты не съели дневной бюджет
MAX_LLM_CALLS = int(os.environ.get("TYRELL_MAX_LLM_CALLS", 150))
LLM_USAGE = {"calls": 0, "prompt": 0, "completion": 0}
KEEP_TOOL_MESSAGES = 3                # столько последних ответов инструментов отдаем LLM целиком
TOOL_MESSAGE_LIMIT = 600              # до скольких символов обрезаем старые ответы
TOOL_RESULT_LIMIT = 4000              # потолок одного ответа инструмента: история растет на каждом шаге
MAX_NUDGES = 3                        # сколько раз просим LLM продолжать, если время осталось
FINAL_MAX_SEEDS = 8                   # сидов на тип модели в финальном auto_train

INSTRUCTION = """Ты — агент для решения задач машинного обучения на табличных данных.
Тебе даны условие, сводка CSV и уже определённый task_spec (target, id, тип задачи, метрика).
task_spec менять нельзя, он передаётся в эксперименты автоматически.

Базовое решение уже обучено и сохранено: auto_train (LightGBM и XGBoost с несколькими
сидами и взвешенным блендом, для метрик по меткам ещё подбор весов классов). Одной моделью
уже проверены автопризнаки (config auto_features: частоты категорий, число пропусков,
части дат, средние и отклонения важных числовых признаков по категориям) и кодирование
по таргету (config target_encoding: среднее таргета по сущности, которая повторяется в
train и test, например зданию или клиенту; считается внутри фолдов без утечки); выигравший вариант обучен
полным рецептом. Смотри в «Уже сделано», какой вариант сохранён, и строй гипотезы поверх него.
Твоя задача — улучшить результат валидации гипотезами. Самые полезные гипотезы:
1. Новые признаки через feature_code: строка Python с функцией make_features(df) -> df.
   df — признаки train и test одной таблицей, без target и id; верни df с теми же строками
   (можно добавлять и удалять колонки). Доступны pd и np. Ищи то, чего нет в автопризнаках:
   отношения и разности числовых признаков, взаимодействия категорий, признаки по смыслу
   полей из условия. Автопризнаки добавляются поверх feature_code, если auto_features true.
2. Параметры модели через config: model (lightgbm, xgboost, catboost), params,
   num_boost_round, early_stopping_rounds, class_weight, log_target, cat_encoding
   (native или frequency), features (список колонок), auto_features (true или false),
   target_encoding (true — ключи найти автоматически, или список исходных колонок; элемент
   списка может быть списком колонок — тогда кодируется их сочетание).
3. recipe: "single" — одна модель (быстро, для проверки гипотезы), "auto" — полный рецепт
   с блендом (медленнее, сильнее). Удачные признаки проверяй потом с recipe "auto".

Валидация — 5 фолдов, одинаковых во всех экспериментах; если test похож на новый период,
фолды строятся группами по этой колонке. Результат сохраняется в submission.csv, только
если он лучше сохранённого на большинстве фолдов и прирост больше шума (поле vs_saved).
Инструмент возвращает score, fold_scores, saved, vs_saved и best_saved_score.
Сравнивай гипотезы по score с одинаковым recipe; разница меньше разброса fold_scores — шум.
В конце решение само запустит расширенный auto_train на лучших найденных признаках.

run_python запускает отдельный процесс для анализа данных; файлы сохраняются, переменные нет.
Относительные пути — от папки задачи. Не устанавливай пакеты, не ходи в интернет,
не читай ключи и .env. Доступно 8 CPU и 32 ГБ памяти.
Работай, пока есть время: после каждого результата предлагай следующую гипотезу.
"""

TOOLS = [{
    "type": "function",
    "function": {
        "name": "run_python",
        "description": "Выполнить Python-код и вернуть вывод, ошибки и код завершения.",
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
            "additionalProperties": False,
        },
    },
}, {
    "type": "function",
    "function": {
        "name": "run_experiment",
        "description": "Обучить модель на текущем task_spec, оценить на фолдах и сохранить submission при улучшении.",
        "parameters": {
            "type": "object",
            "properties": {
                "config": {"type": "object"},
                "recipe": {"type": "string", "enum": ["single", "auto"]},
                "feature_code": {"type": "string"},
                "time_budget": {"type": "number", "minimum": 30, "maximum": 2400},
            },
            "required": ["config", "recipe", "time_budget"],
            "additionalProperties": False,
        },
    },
}]


def log(message):
    print(message, file=sys.stderr, flush=True)


def run_python(code, task_dir, timeout):
    # Окружение наследуем, иначе часть библиотек не находит свои пакеты и HOME,
    # но ключи LLM в код агента не передаем
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("OPENAI_")}
    env |= {
        "PYTHONIOENCODING": "utf-8",
        "PYTHONPATH": str(Path(__file__).resolve().parent),
        "OMP_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "8",
    }
    env.setdefault("HOME", str(task_dir))
    try:
        result = subprocess.run(
            [sys.executable, "-u", "-c", code],
            cwd=task_dir, capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout, env=env,
        )
        return {
            "stdout": result.stdout[-12000:],
            "stderr": result.stderr[-12000:],
            "exit_code": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"error": "Превышено время выполнения кода."}


def experiment(task_dir, spec, budget, recipe="single", config=None, feature_code=None):
    """Эксперимент в отдельном процессе: падение или утечка памяти не задевают агента."""
    payload = json.dumps({"task_spec": spec, "config": config or {}, "time_budget": budget,
                          "recipe": recipe, "feature_code": feature_code}, ensure_ascii=False)
    code = (
        "import json; from pipeline import run_experiment; "
        f"args = json.loads({payload!r}); "
        "print(json.dumps(run_experiment('.', **args), ensure_ascii=False, default=str))"
    )
    timeout = budget * 1.2 + 180
    if HARD_DEADLINE[0] is not None:
        # Не дольше общего дедлайна: сабмит пишется атомарно, недоделанный шаг его не испортит
        timeout = min(timeout, HARD_DEADLINE[0] - time.monotonic())
    if timeout < 30:
        return None, {"error": "Общее время работы истекло."}
    result = run_python(code, task_dir, timeout=timeout)
    if result.get("exit_code") == 0:
        try:
            return json.loads(result["stdout"].strip().splitlines()[-1]), result
        except (ValueError, IndexError):
            pass
    return None, result


def fallback(task_dir, spec):
    """Константный ответ, если не обучилась даже быстрая модель."""
    payload = json.dumps(spec, ensure_ascii=False)
    code = ("import json; from pipeline import write_fallback; "
            f"print(write_fallback('.', json.loads({payload!r})))")
    log(f"Константный ответ: {run_python(code, task_dir, timeout=600)}")
    if not (task_dir / "submission.csv").exists():
        # Последний рубеж: дочерний процесс не смог даже это, пишем из основного
        from pipeline import write_fallback
        try:
            log(f"Константный ответ из основного процесса: {write_fallback(task_dir, spec)}")
        except Exception as error:
            log(f"Не удалось записать даже константный ответ: {error}")


def track_usage(response):
    """Считаем токены: лимит LLM дневной, и по логу видно, сколько стоит прогон."""
    # Поля читаем осторожно: сервер может вернуть usage в другом виде, запрос это не должно ломать
    usage = getattr(response, "usage", None)
    LLM_USAGE["calls"] += 1
    LLM_USAGE["prompt"] += getattr(usage, "prompt_tokens", 0) or 0
    LLM_USAGE["completion"] += getattr(usage, "completion_tokens", 0) or 0
    log(f"LLM: запросов {LLM_USAGE['calls']}, токенов на входе {LLM_USAGE['prompt']}, "
        f"на выходе {LLM_USAGE['completion']}")


def ask_llm(client, messages, deadline):
    """Запрос с повторами. None, если LLM недоступна: агент продолжит без нее."""
    for attempt in range(LLM_ATTEMPTS):
        remaining = deadline - time.monotonic()
        if remaining < 60 or LLM_USAGE["calls"] >= MAX_LLM_CALLS:
            return None
        try:
            response = client.chat.completions.create(
                model=os.environ["OPENAI_MODEL"], messages=trim_history(messages),
                tools=TOOLS, parallel_tool_calls=False, timeout=min(180, remaining - 30),
            )
            track_usage(response)
            return response
        except Exception as error:  # сеть, лимиты, ошибки сервера: пробуем еще раз
            log(f"LLM недоступна (попытка {attempt + 1}): {type(error).__name__}: {error}")
            time.sleep(min(10 * 2 ** attempt, 90, max(0.0, (deadline - time.monotonic()) / 4)))
    return None


def clip_text(text, limit=TOOL_RESULT_LIMIT):
    """Начало и конец длинного ответа: в выводе Python самое важное в конце, ошибки и результат."""
    if len(text) <= limit:
        return text
    head = limit // 4
    return text[:head] + " ...(обрезано)... " + text[-(limit - head):]


def trim_history(messages):
    """Старые ответы инструментов и ассистента обрезаем, чтобы история не тратила лимит.

    Целиком остается все, что после KEEP_TOOL_MESSAGES-го с конца ответа инструмента.
    """
    tool_positions = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    cutoff = tool_positions[-KEEP_TOOL_MESSAGES] if len(tool_positions) >= KEEP_TOOL_MESSAGES else 0
    trimmed = []
    for i, message in enumerate(messages):
        content = message.get("content")
        old = i < cutoff and message.get("role") in ("tool", "assistant")
        if old and isinstance(content, str) and len(content) > TOOL_MESSAGE_LIMIT:
            message = message | {"content": content[:TOOL_MESSAGE_LIMIT] + " ...(обрезано)"}
        trimmed.append(message)
    return trimmed


def compact_summary(summary):
    """Сводка для LLM без описаний колонок: они уже есть в task.md, а токены тратят дневной лимит."""
    out = {k: v for k, v in summary.items() if k not in ("train", "test")}
    out["columns_format"] = "name, dtype, missing, unique"
    for part in ("train", "test"):
        info = summary.get(part, {})
        out[part] = {"rows": info.get("rows"), "columns": [
            [c["name"], c["dtype"], c["missing"], c["unique"]] for c in info.get("columns", [])]}
    return json.dumps(out, ensure_ascii=False)


def resolve_spec(client, task, summary, guess, deadline):
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
    for attempt in range(LLM_ATTEMPTS):
        if deadline - time.monotonic() < 120:
            break
        try:
            response = client.chat.completions.create(
                model=os.environ["OPENAI_MODEL"], messages=messages, timeout=120,
            )
            track_usage(response)
            text = response.choices[0].message.content or ""
            answer = json.loads(text[text.index("{"):text.rindex("}") + 1])
            candidate = spec | {k: answer[k] for k in spec if k in answer}
            if candidate["target_column"] in summary.get("possible_targets", []) + [spec["target_column"]] \
                    and candidate["task_type"] in ("regression", "binary", "multiclass"):
                return candidate
        except Exception as error:
            log(f"Не удалось уточнить task_spec (попытка {attempt + 1}): {error}")
            time.sleep(5)
    return spec


def choose_variant(task_dir, spec, baseline, deadline):
    """Какие улучшения признаков брать в полный рецепт: проверка одной моделью против baseline.

    baseline - страховочная модель (та же одна модель на тех же фолдах). Вариант принимается
    по тому же правилу, что и сабмит: лучше на большинстве фолдов и прирост больше шума.
    Если выиграли оба, проверяем их вместе. Возвращает config победителя или None.
    """
    from training import compare_results

    if not baseline or not baseline.get("complete"):
        return None
    # На одну проверку - с запасом время страховочной модели, но не больше ее доли
    budget = min(SAFETY_SHARE * TOTAL_SECONDS, max(120.0, 3 * baseline["fit_time"]))

    def check(config):
        if deadline - time.monotonic() < FINAL_SHARE * TOTAL_SECONDS + budget * 1.2:
            return None
        report, raw = experiment(task_dir, spec, budget=budget, config=config)
        if report is None:
            log(f"Проверка {config} не удалась: {str(raw)[-500:]}")
            return None
        if not report["complete"]:
            log(f"Проверка {config}: не все фолды успели, сравнить нельзя")
            return None
        result = compare_results(baseline, report)
        log(f"Проверка {config}: {report['score']} против {baseline['score']}, "
            f"принят: {result['accept']}, ключи: {report.get('te_keys')}")
        return result if result["accept"] else None

    variants = [{"auto_features": True}, {"target_encoding": True}]
    accepted = [(config, result) for config in variants if (result := check(config))]
    if len(accepted) == 2:
        both = {"auto_features": True, "target_encoding": True}
        if result := check(both):
            accepted.append((both, result))
    if not accepted:
        return None
    return max(accepted, key=lambda item: item[1]["diff_mean"])[0]


def run_agent(task_path):
    start = time.monotonic()
    deadline = start + TOTAL_SECONDS
    HARD_DEADLINE[0] = deadline
    remaining = lambda: deadline - time.monotonic()
    task_path = task_path.resolve()
    task_dir = task_path.parent
    task = task_path.read_text(encoding="utf-8")
    summary = inspect_data(task_dir)
    (task_dir / ".tyrell_experiments.json").write_text("[]", encoding="utf-8")

    keys_ok = all(os.environ.get(n, "").strip() for n in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"))
    client = OpenAI(max_retries=1) if keys_ok else None
    if not keys_ok:
        log("Ключи LLM не заданы, работаю без LLM.")

    # Шаг 0: task_spec и страховочный ответ до любых долгих запросов к LLM
    try:
        guess = guess_task_spec(task_dir, task)
    except ValueError as error:
        log(f"Эвристика не определила task_spec: {error}")
        guess = ({"target_column": None, "id_columns": [], "task_type": None, "metric": None,
                  "probabilities": False}, False)
    spec = resolve_spec(client, task, summary, guess, deadline)
    log(f"task_spec: {spec}")
    report, raw = experiment(task_dir, spec, budget=SAFETY_SHARE * TOTAL_SECONDS)
    safety = report
    log(f"Страховочный ответ: {report and report['score']}")
    if report is None or not report.get("saved"):
        log(f"Быстрая модель не сохранилась: {str(raw)[-2000:]}")
        fallback(task_dir, spec)

    # Шаг 1: базовый auto_train
    budget = min(AUTO_SHARE * TOTAL_SECONDS, remaining() - FINAL_SHARE * TOTAL_SECONDS)
    if budget > 120:
        report, raw = experiment(task_dir, spec, budget=budget, recipe="auto")
        log(f"auto_train: {report and {k: report[k] for k in ('score', 'saved', 'members')}}")
        if report is None:
            log(f"auto_train упал: {str(raw)[-2000:]}")

    # Шаг 1б: автопризнаки и кодирование по таргету сначала проверяем одной моделью на тех же
    # фолдах - это минуты. Полный рецепт (десятки минут) запускаем только для победителя
    winner = choose_variant(task_dir, spec, safety, deadline)
    budget = min(AUTO_SHARE * TOTAL_SECONDS, remaining() - FINAL_SHARE * TOTAL_SECONDS)
    if winner and budget > 120:
        report, raw = experiment(task_dir, spec, budget=budget, recipe="auto", config=winner)
        log(f"auto_train с {winner}: {report and {k: report[k] for k in ('score', 'saved', 'vs_saved')}}")
        if report is None:
            log(f"auto_train с {winner} упал: {str(raw)[-2000:]}")

    # Шаг 2: цикл гипотез с LLM
    if client is not None:
        run_llm_loop(client, task, summary, spec, task_dir, deadline)

    # Шаг 3: все оставшееся время - расширенный auto_train (больше сидов и медленные модели)
    # на лучших признаках. Сабмит заменится, только если результат лучше с учетом шума
    history = json.loads((task_dir / ".tyrell_experiments.json").read_text(encoding="utf-8"))
    saved = [h for h in history if h["saved"]]
    best = saved[-1] if saved else None
    # Поправка на перерасход auto_train до 5% и загрузку данных в дочернем процессе
    budget = (remaining() - 180) / 1.1
    if budget > 300:
        config = {"max_seeds": FINAL_MAX_SEEDS}
        if best:
            config |= {k: v for k, v in best["config"].items() if k == "cat_encoding"}
            config["auto_features"] = bool(best.get("auto_features"))
            config["target_encoding"] = best.get("target_encoding") or False
        report, _ = experiment(task_dir, spec, budget=budget, recipe="auto", config=config,
                               feature_code=best and best.get("feature_code"))
        log(f"Финальный auto_train: {report and {k: report[k] for k in ('score', 'saved', 'vs_saved')}}")

    try:
        check_result(task_dir)
        history = json.loads((task_dir / ".tyrell_experiments.json").read_text(encoding="utf-8"))
        saved = [h for h in history if h["saved"]]
        print(f"Готово: submission.csv сохранён, лучший результат валидации "
              f"{saved and saved[-1]['score']}.")
    except Exception as error:
        print(f"Проверка submission.csv: {error}")


def run_llm_loop(client, task, summary, spec, task_dir, deadline):
    llm_deadline = deadline - FINAL_SHARE * TOTAL_SECONDS
    history = json.loads((task_dir / ".tyrell_experiments.json").read_text(encoding="utf-8"))
    context = {"task_spec": spec, "experiments": [
        {k: h.get(k) for k in ("recipe", "auto_features", "target_encoding", "te_keys", "score",
                               "fold_scores", "saved", "members")} for h in history]}
    messages = [
        {"role": "system", "content": INSTRUCTION},
        {"role": "user", "content": task + "\n\nСводка данных:\n" + compact_summary(summary)
                                    + "\n\nУже сделано:\n" + json.dumps(context, ensure_ascii=False, default=str)},
    ]
    nudges = 0
    for step in range(200):
        if llm_deadline - time.monotonic() < 120:
            break
        log(f"Шаг {step + 1}. До финала {int(llm_deadline - time.monotonic())} секунд.")
        response = ask_llm(client, messages, llm_deadline)
        if response is None:
            log("LLM не отвечает, перехожу к финалу.")
            return
        message = response.choices[0].message
        reply = message.model_dump(exclude_none=True)
        # Рассуждения модели в историю не кладем: иначе каждый следующий запрос оплачивает их снова
        for key in ("reasoning_content", "reasoning"):
            reply.pop(key, None)
        messages.append(reply)

        if not message.tool_calls:
            left = llm_deadline - time.monotonic()
            if nudges >= MAX_NUDGES or left < 600:
                return
            nudges += 1
            messages.append({"role": "user", "content": f"Осталось около {int(left // 60)} минут. "
                             "Предложи и проверь следующую гипотезу: новые признаки или параметры."})
            continue

        for call in message.tool_calls:
            try:
                arguments = json.loads(call.function.arguments)
                left = llm_deadline - time.monotonic()
                if call.function.name == "run_experiment":
                    budget = min(float(arguments.get("time_budget", 300)), 2400, left - 60)
                    if budget < 30:
                        raise ValueError("Время на новые эксперименты закончилось.")
                    report, raw = experiment(
                        task_dir, spec, budget=budget, recipe=arguments.get("recipe", "single"),
                        config=arguments.get("config") or {}, feature_code=arguments.get("feature_code"),
                    )
                    result = report if report is not None else raw
                elif call.function.name == "run_python":
                    code = arguments["code"]
                    if not isinstance(code, str):
                        raise ValueError("Аргумент code должен быть строкой.")
                    result = run_python(code, task_dir, min(1800, max(30, left)))
                else:
                    raise ValueError("Неизвестный инструмент.")
            except (ValueError, KeyError, TypeError, OSError) as error:
                result = {"error": str(error)}
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": clip_text(json.dumps(result, ensure_ascii=False, default=str)),
            })


if __name__ == "__main__":
    run_agent(Path(sys.argv[1] if len(sys.argv) > 1 else "task.md"))
