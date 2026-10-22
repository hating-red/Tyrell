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
