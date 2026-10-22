"""Сборка, проверка и безопасная запись submission.csv.

Ошибка формата = 0 баллов, какой бы хорошей ни была модель. Поэтому модуль:
- берёт названия столбцов из sample_submission.csv, а порядок id из test.csv;
- чинит то, что можно починить без вреда (пропуски, бесконечности, лишняя размерность);
- не затирает уже записанный хороший файл плохим;
- пишет атомарно: сначала во временный файл, потом подменяет. Если платформа
  остановит процесс на 3-м часу посреди записи, старый файл останется целым.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd


def validate_submission(submission: pd.DataFrame, expected_ids: pd.Series, columns: list[str],
                        allowed_labels: set | None = None) -> list[str]:
    """Проверяет таблицу ответа по правилам task.md. Возвращает список проблем (пустой = всё OK)."""
    if list(submission.columns) != columns:
        return [f"столбцы {list(submission.columns)}, ожидались {columns}"]

    id_col, answer_col = columns
    problems = []
    if len(submission) != len(expected_ids):
        problems.append(f"строк {len(submission)}, ожидалось {len(expected_ids)}")
    elif not np.array_equal(submission[id_col].to_numpy(), expected_ids.to_numpy()):
        problems.append("порядок или состав id не совпадает с test.csv")
    if submission[id_col].duplicated().any():
        problems.append("повторяющиеся id")

    answers = submission[answer_col]
    if answers.isna().any():
        problems.append(f"пропусков в ответах: {int(answers.isna().sum())}")
    if pd.api.types.is_numeric_dtype(answers) and not np.isfinite(answers.dropna()).all():
        problems.append("есть бесконечности в ответах")
    if allowed_labels is not None:
        unknown = set(answers.dropna()) - set(allowed_labels)
        if unknown:
            problems.append(f"классы вне допустимого списка: {sorted(map(str, unknown))[:5]}")
    return problems


def _repair(answers: pd.Series, allowed_labels: set | None) -> tuple[pd.Series, list[str]]:
    """Чинит ответы, если это можно сделать без вреда. Возвращает исправленные ответы и список исправлений."""
    fixes = []
    if pd.api.types.is_numeric_dtype(answers) and allowed_labels is None:
        bad = ~np.isfinite(answers)  # NaN и ±бесконечность
        if bad.any():
            filler = float(answers[~bad].median()) if (~bad).any() else 0.0
            answers = answers.where(~bad, filler)
            fixes.append(f"{int(bad.sum())} пропусков/бесконечностей заменены медианой {filler:.4g}")
    elif answers.isna().any():
        filler = answers.mode().iloc[0] if answers.notna().any() else sorted(allowed_labels or {""})[0]
        fixes.append(f"{int(answers.isna().sum())} пропусков заменены самым частым классом '{filler}'")
        answers = answers.fillna(filler)
    return answers, fixes


def _write_atomic(table: pd.DataFrame, output_path: Path) -> None:
    """Пишет CSV во временный файл рядом и подменяет им старый одним действием (os.replace)."""
    tmp_path = output_path.with_suffix(".tmp")
    table.to_csv(tmp_path, index=False, encoding="utf-8")
    os.replace(tmp_path, output_path)


def make_submission(predictions, test_ids: pd.Series, sample_path: str | Path, output_path: str | Path,
                    allowed_labels: set | None = None) -> list[str]:
    """Собирает submission.csv из предсказаний, проверяет и сохраняет.

    predictions:    одно предсказание на строку test.csv, в том же порядке
    test_ids:       столбец id из test.csv
    sample_path:    sample_submission.csv, из него берём названия столбцов
    output_path:    куда писать (обычно submission.csv рядом с task.md)
    allowed_labels: для классификации — допустимые классы (обычно уникальные значения target в train)

    Возвращает список проблем. Пустой список = файл записан и корректен.
    Если проблемы есть, файл НЕ записывается: старый хороший ответ остаётся на месте.
    """
    columns = list(pd.read_csv(sample_path, nrows=0).columns)
    if len(columns) != 2:
        return [f"в sample_submission {len(columns)} столбцов, модуль рассчитан на 2: {columns}"]

    answers = pd.Series(np.asarray(predictions).reshape(-1))  # (n, 1) → (n,)
    if len(answers) != len(test_ids):
        return [f"предсказаний {len(answers)}, а строк в test {len(test_ids)}"]

    answers, fixes = _repair(answers, allowed_labels)
    for fix in fixes:
        print(f"[submission] исправлено: {fix}", flush=True)

    submission = pd.DataFrame({columns[0]: test_ids.to_numpy(), columns[1]: answers.to_numpy()})
    problems = validate_submission(submission, test_ids.reset_index(drop=True), columns, allowed_labels)
    if problems:
        print(f"[submission] НЕ записан: {'; '.join(problems)}", flush=True)
        return problems

    _write_atomic(submission, Path(output_path))
    print(f"[submission] записан {output_path}: {len(submission)} строк", flush=True)
    return []

