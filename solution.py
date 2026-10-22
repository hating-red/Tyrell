import json
import os
from pathlib import Path
import subprocess
import sys
import time

from openai import OpenAI


INSTRUCTION = """Ты — агент для решения задач машинного обучения.
Прочитай условие и исследуй train.csv, test.csv, data_dictionary.csv
и sample_submission.csv через инструмент run_python.

Определи целевую колонку, тип задачи и метрику из условия и данных.
Раздели обучающие данные на обучение и валидацию с учётом условий задачи.
Обучай предобработку только на обучающей части. Оцени модель на валидации.
Сначала получи простое решение: обучи модель на всех обучающих данных,
предскажи ответы для test.csv и сохрани submission.csv рядом с task.md.
Соблюдай колонки, идентификаторы и порядок строк из sample_submission.csv.

Затем попробуй улучшить результат. Сравнивай модели на одной валидации,
сохраняй результаты экспериментов и оставляй лучший вариант предсказаний.
Не удаляй готовый submission.csv во время новых экспериментов.

Каждый вызов run_python запускает отдельный процесс: переменные не сохраняются,
но файлы сохраняются. Все относительные пути отсчитываются от папки задачи.
Модули проекта доступны для импорта. Используй только установленные библиотеки.
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
    task_path = task_path.resolve()
    messages = [
        {"role": "system", "content": INSTRUCTION},
        {"role": "user", "content": task_path.read_text(encoding="utf-8")},
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
                if not (task_path.parent / "submission.csv").is_file():
                    raise RuntimeError("Агент завершил работу, но не создал submission.csv.")
                print(message.content or "Готово: submission.csv сохранён.")
                return

            for call in message.tool_calls:
                try:
                    if call.function.name != "run_python":
                        raise ValueError("Неизвестный инструмент.")
                    arguments = json.loads(call.function.arguments)
                    code = arguments["code"]
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

    if not (task_path.parent / "submission.csv").is_file():
        raise RuntimeError("Лимит работы достигнут, submission.csv не создан.")
    print("Лимит работы достигнут. Оставлен сохранённый submission.csv.")


if __name__ == "__main__":
    run_agent(Path(sys.argv[1] if len(sys.argv) > 1 else "task.md"))