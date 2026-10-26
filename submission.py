"""Сборка, проверка и безопасная запись submission.csv.

Ошибка формата = 0 баллов, какой бы хорошей ни была модель. Поэтому модуль:
- берёт названия и порядок столбцов из sample_submission.csv, а порядок id из test.csv;
- поддерживает три вида ответа, которые могут встретиться на скрытых задачах:
    1. метка класса или число (регрессия): столбцы `id, target`;
    2. вероятность одного класса (ROC-AUC, LogLoss): `id, target`, значения от 0 до 1;
    3. вероятности по каждому классу: `id, класс_A, класс_B, ...`, сумма по строке = 1;
- чинит то, что можно починить без вреда (пропуски, тип чисел, мелкие выходы за [0, 1]);
- не затирает уже записанный хороший файл плохим;
- пишет атомарно: сначала во временный файл, потом подменяет. Если платформа
  остановит процесс на 3-м часу посреди записи, старый файл останется целым.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

PROBA_TOLERANCE = 1e-6  # выход за [0, 1] меньше этого — ошибка округления, обрезаем молча
ROW_SUM_TOLERANCE = 0.01  # сумма вероятностей в строке может отличаться от 1 не больше чем на столько


# ---------- Проверка ----------

def validate_submission(submission: pd.DataFrame, expected_ids: pd.Series, columns: list[str],
                        allowed_labels: set | None = None, probabilities: bool = False) -> list[str]:
    """Проверяет таблицу ответа по правилам task.md. Возвращает список проблем (пустой = всё OK).

    columns: ожидаемые столбцы в нужном порядке (как в sample_submission.csv)
    allowed_labels: допустимые классы, если ответ — метка класса
    probabilities: True, если ответ — вероятности (один столбец или по столбцу на класс)
    """
    if list(submission.columns) != columns:
        return [f"столбцы {list(submission.columns)}, ожидались {columns}"]

    id_col = expected_ids.name if expected_ids.name in columns else columns[0]
    answer_cols = [c for c in columns if c != id_col]
    problems = []

    if len(submission) != len(expected_ids):
        problems.append(f"строк {len(submission)}, ожидалось {len(expected_ids)}")
    elif not np.array_equal(submission[id_col].to_numpy(), expected_ids.to_numpy()):
        problems.append("порядок или состав id не совпадает с test.csv")
    if submission[id_col].duplicated().any():
        problems.append("повторяющиеся id")

    for col in answer_cols:
        answers = submission[col]
        if answers.isna().any():
            problems.append(f"пропусков в '{col}': {int(answers.isna().sum())}")
        if pd.api.types.is_numeric_dtype(answers) and not np.isfinite(answers.dropna()).all():
            problems.append(f"бесконечности в '{col}'")
        if probabilities and pd.api.types.is_numeric_dtype(answers):
            if ((answers < 0) | (answers > 1)).any():
                problems.append(f"в '{col}' значения вне [0, 1]: возможно, переданы классы вместо вероятностей")
        elif probabilities:
            problems.append(f"в '{col}' ожидались вероятности (числа), а пришёл тип {answers.dtype}")

    if probabilities and len(answer_cols) > 1 and not problems:
        row_sums = submission[answer_cols].sum(axis=1)
        if (row_sums - 1).abs().max() > ROW_SUM_TOLERANCE:
            problems.append(f"сумма вероятностей в строке должна быть 1, а она до {row_sums.max():.3f}")

    if allowed_labels is not None and len(answer_cols) == 1:
        unknown = set(submission[answer_cols[0]].dropna()) - set(allowed_labels)
        if unknown:
            problems.append(f"классы вне допустимого списка: {sorted(map(str, unknown))[:5]}")
    return problems


# ---------- Починка ----------

def _match_dtype(answers: pd.Series, sample_column: pd.Series) -> tuple[pd.Series, list[str]]:
    """Приводит ответы к типу из sample_submission, если это не меняет смысла.

    Пример: в шаблоне классы 0/1 (целые), а модель вернула 0.0/1.0 или True/False.
    Сравнение строк '1' и '1.0' дало бы «неизвестный класс», поэтому приводим к int.
    """
    if not pd.api.types.is_integer_dtype(sample_column):
        return answers, []
    if pd.api.types.is_bool_dtype(answers):
        return answers.astype(int), ["True/False приведены к 1/0, как в шаблоне"]
    if pd.api.types.is_float_dtype(answers):
        finite = answers.dropna()
        if len(finite) and np.allclose(finite, finite.round()):
            return answers.round().astype("Int64"), ["дробные числа вида 1.0 приведены к целым, как в шаблоне"]
    return answers, []


def _repair_single(answers: pd.Series, allowed_labels: set | None, probabilities: bool) -> tuple[pd.Series, list[str]]:
    """Чинит один столбец ответа. Возвращает исправленные ответы и список исправлений."""
    fixes = []
    is_numeric = pd.api.types.is_numeric_dtype(answers) and not pd.api.types.is_bool_dtype(answers)
    if is_numeric and allowed_labels is None:
        bad = ~np.isfinite(answers.astype(float))  # NaN и ±бесконечность
        if bad.any():
            filler = float(answers[~bad].median()) if (~bad).any() else (0.5 if probabilities else 0.0)
            answers = answers.where(~bad, filler)
            fixes.append(f"{int(bad.sum())} пропусков/бесконечностей заменены медианой {filler:.4g}")
        if probabilities:
            tiny = ((answers < 0) & (answers > -PROBA_TOLERANCE)) | ((answers > 1) & (answers < 1 + PROBA_TOLERANCE))
            if tiny.any():
                answers = answers.clip(0, 1).where(tiny, answers)
                fixes.append(f"{int(tiny.sum())} значений чуть за [0, 1] (ошибка округления) обрезаны")
    elif answers.isna().any():
        filler = answers.mode().iloc[0] if answers.notna().any() else sorted(allowed_labels or {""})[0]
        fixes.append(f"{int(answers.isna().sum())} пропусков заменены самым частым классом '{filler}'")
        answers = answers.fillna(filler)
    return answers, fixes


def _repair_class_probabilities(table: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Чинит вероятности по классам: пропуски → равные доли, мелкие отклонения суммы от 1 → нормировка."""
    fixes = []
    if table.isna().any().any():
        rows = table.isna().any(axis=1)
        table.loc[rows] = 1 / table.shape[1]
        fixes.append(f"{int(rows.sum())} строк с пропусками заменены равными вероятностями")
    table = table.clip(lower=0)
    row_sums = table.sum(axis=1)
    close = (row_sums - 1).abs() <= ROW_SUM_TOLERANCE
    if close.all() and not np.allclose(row_sums, 1):
        table = table.div(row_sums, axis=0)
        fixes.append("вероятности в строках нормированы, чтобы сумма была ровно 1")
    return table, fixes


# ---------- Сборка и запись ----------

def _write_atomic(table: pd.DataFrame, output_path: Path) -> None:
    """Пишет CSV во временный файл рядом и подменяет им старый одним действием (os.replace)."""
    tmp_path = output_path.with_suffix(".tmp")
    table.to_csv(tmp_path, index=False, encoding="utf-8")
    os.replace(tmp_path, output_path)


def _find_id_column(columns: list[str], test_ids: pd.Series) -> str:
    """Ищет столбец id по названию: имя из test.csv → столбец 'id' → первый столбец."""
    if test_ids.name in columns:
        return test_ids.name
    lowered = {c.lower(): c for c in columns}
    return lowered.get("id", columns[0])


def make_submission(predictions, test_ids: pd.Series, sample_path: str | Path, output_path: str | Path,
                    allowed_labels: set | None = None, probabilities: bool = False,
                    class_order: list | None = None) -> list[str]:
    """Собирает submission.csv из предсказаний, проверяет и сохраняет.

    predictions:    ответы по строкам test.csv в том же порядке. Варианты:
                    - список или массив (n,): метка класса, число или вероятность;
                    - массив (n, k): вероятности по k классам (например, model.predict_proba);
                    - DataFrame с названиями классов в столбцах.
    test_ids:       столбец id из test.csv (pd.Series)
    sample_path:    sample_submission.csv: из него берём названия, порядок и типы столбцов
    output_path:    куда писать (обычно submission.csv рядом с task.md)
    allowed_labels: для меток класса — допустимые классы (обычно уникальные значения target в train)
    probabilities:  True, если ответ — вероятности (метрики ROC-AUC, LogLoss и т.п.)
    class_order:    для массива (n, k): какому классу соответствует каждый столбец
                    (например, model.classes_). Без него считаем, что порядок как в шаблоне.

    Возвращает список проблем. Пустой список = файл записан и корректен.
    Если проблемы есть, файл НЕ записывается: старый хороший ответ остаётся на месте.
    """
    sample = pd.read_csv(sample_path)
    columns = list(sample.columns)
    id_col = _find_id_column(columns, test_ids)
    answer_cols = [c for c in columns if c != id_col]
    if len(test_ids) == 0:
        return ["test пустой: нечего записывать"]

    fixes = []
    if len(answer_cols) == 1:
        values = np.asarray(predictions)
        if values.ndim == 2 and values.shape[1] == 1:
            values = values.reshape(-1)  # (n, 1) → (n,)
        if values.ndim != 1:
            return [f"шаблон ждёт один столбец ответа, а пришло предсказаний формы {values.shape}"]
        answers, fixes = _repair_single(pd.Series(values), allowed_labels, probabilities)
        answers, dtype_fixes = _match_dtype(answers, sample[answer_cols[0]])
        fixes += dtype_fixes
        answer_table = pd.DataFrame({answer_cols[0]: answers.to_numpy()})
    else:
        # Вероятности по классам: выстраиваем столбцы в порядке шаблона
        if isinstance(predictions, pd.DataFrame):
            missing = [c for c in answer_cols if c not in predictions.columns]
            if missing:
                return [f"в предсказаниях нет столбцов для классов {missing[:5]}"]
            answer_table = predictions[answer_cols].reset_index(drop=True).astype(float)
        else:
            values = np.asarray(predictions, dtype=float)
            if values.ndim != 2 or values.shape[1] != len(answer_cols):
                return [f"шаблон ждёт {len(answer_cols)} столбцов вероятностей, а пришло форма {values.shape}"]
            names = [str(c) for c in class_order] if class_order is not None else answer_cols
            if sorted(names) != sorted(answer_cols):
                return [f"class_order {names[:5]} не совпадает с классами шаблона {answer_cols[:5]}"]
            answer_table = pd.DataFrame(values, columns=names)[answer_cols]
        answer_table, fixes = _repair_class_probabilities(answer_table)
        probabilities = True

    if len(answer_table) != len(test_ids):
        return [f"предсказаний {len(answer_table)}, а строк в test {len(test_ids)}"]
    for fix in fixes:
        print(f"[submission] исправлено: {fix}", flush=True)

    answer_table.insert(0, id_col, test_ids.to_numpy())
    submission = answer_table[columns]  # порядок столбцов как в шаблоне
    expected_ids = test_ids.reset_index(drop=True).rename(id_col)
    problems = validate_submission(submission, expected_ids, columns, allowed_labels, probabilities)
    if problems:
        print(f"[submission] НЕ записан: {'; '.join(problems)}", flush=True)
        return problems

    _write_atomic(submission, Path(output_path))
    print(f"[submission] записан {output_path}: {len(submission)} строк, столбцы {columns}", flush=True)
    return []
