"""Тренажёр: проверяет solution.py так, как это сделает платформа, но с ответами, которые знаем мы.

Что делает:
1. Отрезает от train «скрытый тест» (его ответы solution.py не увидит).
2. Собирает из этого копию задачи в runs/ с той же структурой папок, что на платформе.
3. Запускает `python solution.py task.md` с лимитом времени и ключами из .env.
4. Проверяет формат submission.csv и считает метрику по отрезанным ответам.
5. Дописывает строку в runs/history.csv, чтобы сравнивать версии между собой.

Пример:
    python -m trainer.run_local data/tasks/regression --holdout-col Year_Factor
    python -m trainer.run_local data/tasks/multiclass --time-limit 600
"""

import argparse
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import dotenv_values
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split

from submission import validate_submission

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"
HOLDOUT_SHARE = 0.2
RANDOM_STATE = 42

# Метрики: функция и направление (True = больше лучше)
METRICS = {
    "rmse": (lambda y, p: float(np.sqrt(mean_squared_error(y, p))), False),
    "mae": (lambda y, p: float(mean_absolute_error(y, p)), False),
    "r2": (lambda y, p: float(r2_score(y, p)), True),
    "f1_macro": (lambda y, p: float(f1_score(y, p, average="macro")), True),
    "f1_weighted": (lambda y, p: float(f1_score(y, p, average="weighted")), True),
    "accuracy": (lambda y, p: float(accuracy_score(y, p)), True),
    "roc_auc": (lambda y, p: float(roc_auc_score(y, p)), True),
}

# Как метрика называется в тексте task.md. Порядок важен: Macro F1 проверяем раньше просто F1.
METRIC_PATTERNS = [
    ("f1_macro", r"macro[\s_-]*f1|f1[\s_-]*macro|average=\"macro\""),
    ("f1_weighted", r"weighted[\s_-]*f1"),
    ("rmse", r"\brmse\b"),
    ("mae", r"\bmae\b"),
    ("roc_auc", r"roc[\s_-]*auc"),
    ("r2", r"\br2\b|r²"),
    ("accuracy", r"\baccuracy\b"),
]


def read_roles(dictionary_path: Path) -> dict[str, list[str]]:
    """Столбцы из data_dictionary.csv по ролям: {"id": [...], "target": [...], "feature": [...]}."""
    dictionary = pd.read_csv(dictionary_path)
    return dictionary.groupby("role")["feature"].apply(list).to_dict()


def detect_metric(task_text: str) -> str:
    """Находит основную метрику по тексту task.md: первое совпадение по списку шаблонов."""
    lowered = task_text.lower()
    for name, pattern in METRIC_PATTERNS:
        if re.search(pattern, lowered):
            return name
    raise ValueError("Не удалось определить метрику по task.md, укажите --metric")


def split_holdout(train: pd.DataFrame, target: str, holdout_col: str | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Делит train на «новый train» и «скрытый тест».

    holdout_col: если тест на платформе «из будущего» (как год 6 в регрессии), отрезаем
    последнее значение этого столбца. Иначе — случайные 20%, для классификации с сохранением долей классов.
    """
    if holdout_col:
        last = train[holdout_col].max()
        return train[train[holdout_col] < last], train[train[holdout_col] == last]
    is_classification = not pd.api.types.is_float_dtype(train[target])
    stratify = train[target] if is_classification else None
    return train_test_split(train, test_size=HOLDOUT_SHARE, random_state=RANDOM_STATE, stratify=stratify)


def build_sandbox(task_dir: Path, run_dir: Path, holdout_col: str | None) -> tuple[Path, pd.DataFrame, str, str]:
    """Копирует задачу в run_dir, подменяя train/test/sample_submission на версии с отрезанным тестом."""
    shutil.copytree(task_dir, run_dir)
    data_dir = next(run_dir.rglob("train.csv")).parent

    roles = read_roles(data_dir / "data_dictionary.csv")
    target, id_col = roles["target"][0], roles["id"][0]

    full_train = pd.read_csv(data_dir / "train.csv")
    new_train, hidden = split_holdout(full_train, target, holdout_col)

    new_train.to_csv(data_dir / "train.csv", index=False)
    hidden.drop(columns=[target]).to_csv(data_dir / "test.csv", index=False)
    sample = pd.DataFrame({id_col: hidden[id_col], target: new_train[target].iloc[0]})
    sample.to_csv(data_dir / "sample_submission.csv", index=False)

    print(f"[тренажёр] train: {len(new_train)} строк, скрытый тест: {len(hidden)} строк", flush=True)
    return run_dir / "task.md", hidden[[id_col, target]], target, id_col


def check_submission(path: Path, hidden: pd.DataFrame, target: str, id_col: str) -> list[str]:
    """Проверяет файл теми же правилами, что и submission.py (одна проверка на весь проект)."""
    if not path.exists():
        return ["submission.csv не создан"]
    allowed = None if pd.api.types.is_numeric_dtype(hidden[target]) else set(hidden[target])
    return validate_submission(pd.read_csv(path), hidden[id_col].reset_index(drop=True), [id_col, target], allowed)


def git_version() -> str:
    """Короткий хеш текущего коммита и пометка, если есть незакоммиченные изменения."""
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "."], capture_output=True, text=True, cwd=ROOT).stdout
    return commit + ("+изменения" if dirty else "")


def main() -> None:
    parser = argparse.ArgumentParser(description="Локальная проверка solution.py")
    parser.add_argument("task_dir", type=Path, help="каталог с task.md и данными")
    parser.add_argument("--solution", type=Path, default=ROOT / "solution.py")
    parser.add_argument("--holdout-col", help="столбец времени: отрезать его последнее значение как тест")
    parser.add_argument("--metric", choices=METRICS, help="если не указана, ищется в task.md")
    parser.add_argument("--time-limit", type=int, default=3 * 60 * 60, help="секунды, как на платформе: 10800")
    parser.add_argument("--note", default="", help="заметка в журнал: что поменяли в этой версии")
    args = parser.parse_args()

    if not args.solution.exists():
        sys.exit(f"[тренажёр] не найден {args.solution}. Укажите путь к решению через --solution")
    task_dir = args.task_dir.resolve()
    metric = args.metric or detect_metric((task_dir / "task.md").read_text(encoding="utf-8"))
    run_dir = RUNS_DIR / f"{task_dir.name}_{datetime.now():%Y%m%d_%H%M%S}"
    task_md, hidden, target, id_col = build_sandbox(task_dir, run_dir, args.holdout_col)

    # Окружение как на платформе: ключи LLM из .env
    env = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", **{k: v for k, v in dotenv_values(ROOT / ".env").items() if v}}
    print(f"[тренажёр] запускаю {args.solution.name}, лимит {args.time_limit} с, метрика {metric}", flush=True)
    start = time.time()
    try:
        result = subprocess.run([sys.executable, str(args.solution.resolve()), str(task_md)],
                                cwd=args.solution.resolve().parent, env=env, timeout=args.time_limit)
        status = "ok" if result.returncode == 0 else f"упал (код {result.returncode})"
    except subprocess.TimeoutExpired:
        status = "превышен лимит времени"
    elapsed = time.time() - start

    problems = check_submission(task_md.parent / "submission.csv", hidden, target, id_col)
    score = None
    if not problems:
        submission = pd.read_csv(task_md.parent / "submission.csv")
        score_fn, _ = METRICS[metric]
        score = round(score_fn(hidden[target].values, submission[target].values), 4)

    print("\n========== РЕЗУЛЬТАТ ==========")
    print(f"статус: {status}, время: {elapsed:.0f} с")
    print(f"формат: {'OK' if not problems else '; '.join(problems)}")
    print(f"{metric}: {score if score is not None else '—'} ({'больше' if METRICS[metric][1] else 'меньше'} = лучше)")

    history = RUNS_DIR / "history.csv"
    row = pd.DataFrame([{
        "time": f"{datetime.now():%Y-%m-%d %H:%M}", "version": git_version(), "task": task_dir.name,
        "holdout": args.holdout_col or "random", "metric": metric, "score": score, "status": status,
        "seconds": round(elapsed), "format_ok": not problems, "note": args.note,
    }])
    row.to_csv(history, mode="a", header=not history.exists(), index=False)
    print(f"записано в {history.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
