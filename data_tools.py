import csv
from pathlib import Path

import pandas as pd


FILES = ("train.csv", "test.csv", "data_dictionary.csv", "sample_submission.csv")


def _read_files(task_dir):
    task_dir = Path(task_dir).resolve()
    folders = [task_dir] if all((task_dir / name).is_file() for name in FILES) else [
        path.parent for path in task_dir.rglob("train.csv")
        if all((path.parent / name).is_file() for name in FILES)
    ]
    if len(folders) != 1:
        raise ValueError("Нужен один набор train/test/data_dictionary/sample_submission в папке задачи.")
    folder = folders[0]
    headers = {}
    for name in FILES:
        with (folder / name).open(encoding="utf-8-sig", newline="") as stream:
            columns = next(csv.reader(stream), [])
        if not columns or len(set(columns)) != len(columns) or any(not c.strip() for c in columns):
            raise ValueError(f"Пустые или повторяющиеся названия колонок в {name}.")
        headers[name] = columns

    dictionary = pd.read_csv(folder / "data_dictionary.csv", encoding="utf-8-sig").fillna("")
    if not {"feature", "role"}.issubset(dictionary.columns):
        raise ValueError("В data_dictionary.csv нужны колонки feature и role.")
    roles = {
        str(row["feature"]): str(row["role"]).strip().lower()
        for row in dictionary.to_dict("records")
    }
    id_columns = {name for name, role in roles.items() if role in ("id", "identifier")}
    if not id_columns:
        id_columns = set(headers["test.csv"]) & set(headers["sample_submission.csv"])
    frames = {}
    for name in ("train.csv", "test.csv", "sample_submission.csv"):
        frames[name] = pd.read_csv(
            folder / name, encoding="utf-8-sig",
            dtype={col: str for col in id_columns if col in headers[name]},
        )
    return folder, frames["train.csv"], frames["test.csv"], frames["sample_submission.csv"], dictionary


def _candidates(train, test, sample, dictionary):
    roles = {
        str(row["feature"]): str(row["role"]).strip().lower()
        for row in dictionary.to_dict("records")
    }
    targets = [name for name in train if roles.get(name) in ("target", "label")]
    targets += [name for name in train if name not in test]
    ids = [name for name in test if name in sample and roles.get(name) in ("id", "identifier")]
    if not ids:
        ids = [name for name in sample if name in test and test[name].is_unique]
    return list(dict.fromkeys(targets)), list(dict.fromkeys(ids))


def inspect_data(task_dir) -> dict:
    folder, train, test, sample, dictionary = _read_files(task_dir)
    targets, ids = _candidates(train, test, sample, dictionary)
    descriptions = {
        str(row["feature"]): {str(k): str(v)[:300] for k, v in row.items() if k != "feature"}
        for row in dictionary.to_dict("records")
    }
    summary = {
        "data_dir": str(folder),
        "possible_targets": targets,
        "possible_id_columns": ids,
        "sample_columns": list(sample.columns),
        "sample_rows": len(sample),
        "warnings": [],
    }
    for name, frame in (("train", train), ("test", test)):
        summary[name] = {
            "rows": len(frame),
            "columns": [{
                "name": col,
                "dtype": str(frame[col].dtype),
                "missing": int(frame[col].isna().sum()),
                "unique": int(frame[col].nunique()),
                "dictionary": descriptions.get(col, {}),
            } for col in frame],
        }
    summary["targets"] = {}
    for name in targets:
        values = train[name].dropna()
        summary["targets"][name] = {
            "dtype": str(values.dtype),
            "unique": int(values.nunique()),
            "examples": [str(v)[:100] for v in values.unique()[:10]],
        }
    if len(sample) != len(test):
        summary["warnings"].append("Количество строк sample_submission и test различается.")
    if len(targets) != 1:
        summary["warnings"].append("Нужно явно выбрать целевую колонку по условию задачи.")
    features = [c for c in train if c in test and c not in targets and c not in ids]
    summary["warnings"] += data_warnings(train, test, features, targets[0] if len(targets) == 1 else None)
    split = detect_split(train[features], test[features])
    summary["validation_split"] = split
    return summary


def load_data(task_dir, task_spec) -> dict:
    folder, train, test, sample, dictionary = _read_files(task_dir)
    targets, ids = _candidates(train, test, sample, dictionary)
    target = task_spec.get("target_column")
    if target is None and len(targets) == 1:
        target = targets[0]
    if target not in train or target in test:
        raise ValueError("Укажите target_column: он должен быть в train и отсутствовать в test.")

    id_columns = task_spec.get("id_columns")
    if id_columns is None:
        identifier = task_spec.get("id_column")
        id_columns = [identifier] if identifier else ids
    if not isinstance(id_columns, list) or len(id_columns) != 1:
        raise ValueError("Текущий submission.py поддерживает ровно одну ID-колонку; укажите id_columns.")
    identifier = id_columns[0]
    if identifier not in test or identifier not in sample:
        raise ValueError("ID-колонка должна присутствовать в test и sample_submission.")
    if test[identifier].isna().any() or not test[identifier].is_unique:
        raise ValueError("ID в test должны быть заполненными и уникальными.")
    if len(test) == 0 or len(test) != len(sample):
        raise ValueError("test и sample_submission должны иметь одинаковое ненулевое число строк.")

    declared_ids = {
        str(row["feature"]) for row in dictionary.to_dict("records")
        if str(row["role"]).strip().lower() in ("id", "identifier")
    }
    excluded = {target, identifier, *targets, *declared_ids}
    features = task_spec.get("feature_columns")
    if features is None:
        features = [name for name in train if name in test and name not in excluded]
    if not isinstance(features, list) or not features or len(set(features)) != len(features):
        raise ValueError("Нужен непустой список различных признаков feature_columns.")
    if any(name not in train or name not in test or name in excluded for name in features):
        raise ValueError("Признаки должны быть в train и test и не содержать target или ID.")
    if train[target].notna().sum() < 10:
        raise ValueError("Для training.py нужно хотя бы 10 строк с заполненным target.")

    return {
        "X_train": train[features].copy(),
        "y_train": train[target].copy(),
        "X_test": test[features].copy(),
        "test_ids": test[identifier].copy(),
        "sample_path": folder / "sample_submission.csv",
        "target_column": target,
        "id_column": identifier,
        "feature_columns": features,
    }


def detect_split(X_train, X_test, max_groups=50, min_unseen=0.9):
    """Колонка для групповой валидации: значения в test почти все новые для train.

    Типичный случай - код периода (год 6 в test при годах 1-5 в train). Тогда случайные
    фолды завышают оценку, а группа целиком в одном фолде похожа на test.
    Возвращает {"type": "group", "column": имя} или None.
    """
    best, best_share = None, 0.0
    for name in X_test.columns:
        if name not in X_train:
            continue
        n_groups = X_train[name].nunique()
        if not 2 <= n_groups <= max_groups:
            continue
        values = X_test[name].dropna()
        if values.empty:
            continue
        share = float((~values.isin(set(X_train[name].dropna()))).mean())
        if share >= min_unseen and share > best_share:
            best, best_share = name, share
    return {"type": "group", "column": best} if best else None


def data_warnings(train, test, features, target=None):
    """Короткие предупреждения о данных для LLM: сдвиг, новые категории, утечки, константы."""
    warnings = []
    for name in features:
        column, test_column = train[name], test[name]
        if column.nunique(dropna=False) <= 1:
            warnings.append(f"'{name}' константа в train, бесполезна для модели.")
            continue
        is_text = not pd.api.types.is_numeric_dtype(column)
        if is_text:
            unseen = float((test_column.notna() & ~test_column.isin(set(column.dropna()))).mean())
            if unseen >= 0.01:
                warnings.append(f"'{name}': {unseen:.0%} строк test с категориями, которых нет в train.")
            if column.nunique() > 0.5 * len(column):
                warnings.append(f"'{name}': почти уникальна в каждой строке, похоже на id или текст.")
        if target is not None and not is_text and pd.api.types.is_numeric_dtype(train[target]):
            corr = column.corr(train[target])
            if pd.notna(corr) and abs(corr) > 0.98:
                warnings.append(f"'{name}' почти совпадает с таргетом (корреляция {corr:.3f}), возможна утечка.")
    split = detect_split(train[features], test[features])
    if split:
        warnings.append(f"Значения '{split['column']}' в test новые для train: валидация идет "
                        f"группами по этой колонке, как будущий период.")
    return warnings


# Шаблоны метрик в тексте task.md. Порядок важен: частные случаи раньше общих
METRIC_PATTERNS = [
    ("macro_f1", r"macro[\s_-]*f1|f1[\s_-]*macro|average\s*=\s*[\"']macro"),
    ("weighted_f1", r"weighted[\s_-]*f1|f1[\s_-]*weighted"),
    ("rmsle", r"\brmsle\b|root mean squared log"),
    ("rmse", r"\brmse\b|root mean squared error"),
    ("mse", r"\bmse\b|mean squared error"),
    ("mae", r"\bmae\b|mean absolute error"),
    ("mape", r"\bmape\b"),
    ("r2", r"\br2\b|r²|r\^2"),
    ("pr_auc", r"pr[\s_-]*auc|average[\s_-]*precision"),
    ("gini", r"\bgini\b|джини"),
    ("roc_auc", r"roc[\s_-]*auc|\bauc\b"),
    ("logloss", r"log[\s_-]*loss|cross[\s_-]*entropy"),
    ("balanced_accuracy", r"balanced[\s_-]*accuracy"),
    ("mcc", r"\bmcc\b|matthews"),
    ("f1", r"\bf1\b"),
    ("accuracy", r"\baccuracy\b|доля верных"),
]
PROBA_METRICS = {"roc_auc", "gini", "pr_auc", "logloss"}
REGRESSION_METRICS = {"rmse", "rmsle", "mse", "mae", "mape", "r2"}


def detect_metric(task_text):
    """Основная метрика по тексту: сначала строки со словами «основная метрика» или «оценки»."""
    import re

    lowered = task_text.lower()
    focus = [line for line in lowered.splitlines()
             if "метрик" in line and ("основн" in line or "оценк" in line)]
    for text in ("\n".join(focus), lowered):
        for name, pattern in METRIC_PATTERNS:
            if re.search(pattern, text):
                return name
    return None


def guess_task_spec(task_dir, task_text=""):
    """task_spec без LLM: таргет и id из data_dictionary, метрика из task.md, тип по данным.

    Возвращает (spec, confident). confident=False, если таргет или метрику пришлось угадывать.
    """
    folder, train, test, sample, dictionary = _read_files(task_dir)
    targets, ids = _candidates(train, test, sample, dictionary)
    confident = len(targets) == 1 and bool(ids)
    if not targets or not ids:
        raise ValueError("Не удалось найти таргет или id по данным.")
    target, identifier = targets[0], ids[0]
    y = train[target].dropna()

    metric = detect_metric(task_text)
    confident = confident and metric is not None
    if metric is not None:
        is_regression = metric in REGRESSION_METRICS
    else:
        is_integer = pd.api.types.is_numeric_dtype(y) and (y.round() == y).all()
        is_regression = pd.api.types.is_numeric_dtype(y) and not (is_integer and y.nunique() <= 20)
    if is_regression:
        task_type = "regression"
    else:
        task_type = "binary" if y.nunique() == 2 else "multiclass"
    if metric is None or (metric in REGRESSION_METRICS) != is_regression:
        metric = {"regression": "rmse", "binary": "roc_auc", "multiclass": "macro_f1"}[task_type]
    if metric in ("f1", "gini", "pr_auc") and task_type == "multiclass":
        metric = "macro_f1"
    spec = {
        "target_column": target,
        "id_columns": [identifier],
        "task_type": task_type,
        "metric": metric,
        "probabilities": metric in PROBA_METRICS or len(sample.columns) > 2,
    }
    return spec, confident
