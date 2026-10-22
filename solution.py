import json
import os
from pathlib import Path
import subprocess
import sys
import time

from openai import OpenAI

from data_tools import inspect_data
from pipeline import check_result


INSTRUCTION = """Ты — агент для решения задач машинного обучения.
Тебе даны условие и сводка CSV от data_tools.inspect_data.
Определи целевую колонку, ID, тип задачи и ОСНОВНУЮ метрику по условию.
Не выбирай тип задачи только по числовому типу target: числа бывают метками классов.

Вызывай run_experiment: он соединяет data_tools.load_data,
training.train_and_predict и submission.make_submission.
task_spec содержит target_column, id_columns (список из одной колонки),
task_type (regression, binary или multiclass), metric, probabilities (нужны ли
вероятности В ФАЙЛЕ ответа). Для бинарных вероятностей можно задать positive_class.
Метрики: rmse, mse, mae, r2, rmsle, mape, macro_f1, weighted_f1, f1,
accuracy, balanced_accuracy, mcc, roc_auc, gini, pr_auc, logloss.
config задаёт model (lightgbm, catboost, xgboost), params, num_boost_round,
early_stopping_rounds, class_weight, features. Начни с простой быстрой модели.
Валидация — одинаковые случайные фолды с фиксированным seed; для классификации
они стратифицированы. Не считай такую оценку подходящей для временного/группового
разбиения, если оно требуется условием; сообщи об этом ограничении.

Сначала сохрани базовый submission, затем предложи гипотезу улучшения и запусти
следующий эксперимент. Инструмент возвращает score, saved и problems.
Чем лучше метрика, тем лучше эксперимент; направление возвращается инструментом.
Файл перезаписывается только при улучшении сравнимого результата валидации.
Не меняй task_spec между экспериментами. Анализируй ошибки и исправляй config.
Не пиши submission.csv вручную: сборкой и проверкой занимается submission.py.

Каждый вызов run_python запускает отдельный процесс: переменные не сохраняются,
но файлы сохраняются. Все относительные пути отсчитываются от папки задачи.
run_python нужен для дополнительного анализа и генерации признаков.
Для обучения используй run_experiment. Модули проекта доступны для импорта.
Используй только установленные библиотеки.
Не устанавливай пакеты, не обращайся в интернет и не читай ключи или .env.
Доступно 8 CPU и 32 ГБ памяти. На один запуск кода — до 30 минут,
на всю задачу — чуть меньше 3 часов. Сначала сохрани базовые предсказания.
В конце кратко сообщи результат валидации и путь к submission.csv.
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
}]

TOOLS.append({
    "type": "function",
    "function": {
        "name": "run_experiment",
        "description": "Загрузить данные, обучить модель, оценить и сохранить лучший submission.",
        "parameters": {
            "type": "object",
            "properties": {
                "task_spec": {
                    "type": "object",
                    "properties": {
                        "target_column": {"type": "string"},
                        "id_columns": {"type": "array", "items": {"type": "string"}},
                        "task_type": {"type": "string", "enum": ["regression", "binary", "multiclass"]},
                        "metric": {"type": "string"},
                        "probabilities": {"type": "boolean"},
                        "positive_class": {"type": ["string", "number", "boolean"]},
                    },
                    "required": ["target_column", "id_columns", "task_type", "metric", "probabilities"],
                    "additionalProperties": False,
                },
                "config": {"type": "object"},
                "time_budget": {"type": "number", "minimum": 10, "maximum": 1500},
            },
            "required": ["task_spec", "config", "time_budget"],
            "additionalProperties": False,
        },
    },
})


def run_python(code, task_dir, timeout):
    try:
        result = subprocess.run(
            [sys.executable, "-u", "-c", code],
            cwd=task_dir,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env={
                "PATH": os.defpath,
                "PYTHONIOENCODING": "utf-8",
                "PYTHONPATH": str(Path(__file__).resolve().parent),
                "OMP_NUM_THREADS": "8",
                "OPENBLAS_NUM_THREADS": "8",
            },
        )
        return {
            "stdout": result.stdout[-12000:],
            "stderr": result.stderr[-12000:],
            "exit_code": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"error": "Превышено время выполнения кода."}


def run_agent(task_path):
    deadline = time.monotonic() + 3 * 60 * 60 - 60
    for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"):
        if not os.environ.get(name, "").strip():
            raise ValueError(f"Задайте переменную окружения {name} перед запуском.")
    task_path = task_path.resolve()
    task = task_path.read_text(encoding="utf-8")
    summary = inspect_data(task_path.parent)
    (task_path.parent / ".tyrell_experiments.json").write_text("[]", encoding="utf-8")
    messages = [
        {"role": "system", "content": INSTRUCTION},
        {"role": "user", "content": task + "\n\nСводка данных:\n" + json.dumps(summary, ensure_ascii=False)},
    ]

    with OpenAI(max_retries=0) as client:
        for step in range(100):
            remaining = deadline - time.monotonic()
            if remaining <= 1:
                break
            print(f"Шаг {step + 1}. Осталось {int(remaining)} секунд.", file=sys.stderr)
            response = client.chat.completions.create(
                model=os.environ["OPENAI_MODEL"],
                messages=messages,
                tools=TOOLS,
                parallel_tool_calls=False,
                timeout=min(120, remaining),
            )
            message = response.choices[0].message
            messages.append(message.model_dump(exclude_none=True))

            if not message.tool_calls:
                check_result(task_path.parent)
                print(message.content or "Готово: submission.csv сохранён.")
                return

            for call in message.tool_calls:
                try:
                    arguments = json.loads(call.function.arguments)
                    remaining = deadline - time.monotonic()
                    if call.function.name == "run_experiment":
                        budget = min(float(arguments.get("time_budget", 300)), 1500, remaining - 5)
                        if budget <= 0:
                            raise ValueError("Время на новый эксперимент закончилось.")
                        payload = json.dumps({
                            "task_spec": arguments["task_spec"],
                            "config": arguments.get("config", {}), "time_budget": budget,
                        }, ensure_ascii=False)
                        code = (
                            "import json; from pipeline import run_experiment; "
                            f"args = json.loads({payload!r}); "
                            "print(json.dumps(run_experiment('.', **args), ensure_ascii=False))"
                        )
                    elif call.function.name == "run_python":
                        code = arguments["code"]
                    else:
                        raise ValueError("Неизвестный инструмент.")
                    if not isinstance(code, str):
                        raise ValueError("Аргумент code должен быть строкой.")
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        result = {"error": "Общее время работы истекло."}
                    else:
                        result = run_python(code, task_path.parent, min(1800, remaining))
                except (ValueError, KeyError, TypeError, OSError) as error:
                    result = {"error": str(error)}
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result, ensure_ascii=False),
                })

    check_result(task_path.parent)
    print("Лимит работы достигнут. Оставлен сохранённый submission.csv.")


if __name__ == "__main__":
    run_agent(Path(sys.argv[1] if len(sys.argv) > 1 else "task.md"))
