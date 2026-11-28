import csv
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


FILES = ("train.csv", "test.csv", "sample_submission.csv")


def _read_files(task_dir):
    task_dir = Path(task_dir).resolve()
    folders = [task_dir] if all((task_dir / name).is_file() for name in FILES) else [
        path.parent for path in task_dir.rglob("train.csv")
        if all((path.parent / name).is_file() for name in FILES)
    ]
    if len(folders) != 1:
        raise ValueError("Нужен один набор train/test/sample_submission в папке задачи.")
    folder = folders[0]
    headers = {}
    dictionary_path = folder / "data_dictionary.csv"
    file_names = (*FILES, "data_dictionary.csv") if dictionary_path.is_file() else FILES
    for name in file_names:
        with (folder / name).open(encoding="utf-8-sig", newline="") as stream:
            columns = next(csv.reader(stream), [])
        if not columns or len(set(columns)) != len(columns) or any(not c.strip() for c in columns):
            raise ValueError(f"Пустые или повторяющиеся названия колонок в {name}.")
        headers[name] = columns

    dictionary = (pd.read_csv(dictionary_path, encoding="utf-8-sig").fillna("")
                  if dictionary_path.is_file() else pd.DataFrame(columns=["feature", "role"]))
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
            converters={col: str for col in id_columns if col in headers[name]},
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
    if (test[identifier].isna().any() or test[identifier].astype(str).str.strip().eq("").any()
            or not test[identifier].is_unique):
        raise ValueError("ID в test должны быть заполненными и уникальными.")
    if len(test) == 0 or len(test) != len(sample):
        raise ValueError("test и sample_submission должны иметь одинаковое ненулевое число строк.")

    declared_ids = {
        str(row["feature"]) for row in dictionary.to_dict("records")
        if str(row["role"]).strip().lower() in ("id", "identifier")
    }
    excluded = {target, identifier, *targets, *declared_ids, *task_spec.get("excluded_features", ())}
    features = task_spec.get("feature_columns")
    if features is None:
        features = [name for name in train if name in test and name not in excluded]
    if not isinstance(features, list) or len(set(features)) != len(features):
        raise ValueError("Нужен список различных признаков feature_columns.")
    if any(name not in train or name not in test or name in excluded for name in features):
        raise ValueError("Признаки должны быть в train и test и не содержать target или ID.")
    if not train[target].notna().any():
        raise ValueError("В train отсутствуют заполненные значения target.")

    return {
        "X_train": train[features].copy(),
        "y_train": train[target].copy(),
        "train_ids": train[identifier].copy() if identifier in train else None,
        "X_test": test[features].copy(),
        "test_ids": test[identifier].copy(),
        "sample_path": folder / "sample_submission.csv",
        "target_column": target,
        "id_column": identifier,
        "feature_columns": features,
    }


# Event clocks take precedence over metadata dates such as when a customer registered.
_EVENT_CLOCKS = (
    "event_timestamp", "event_time", "event_ts", "call_timestamp", "call_time", "call_ts",
    "incident_timestamp", "incident_time", "request_timestamp", "request_time",
    "occurred_at", "called_at", "event_datetime", "call_datetime", "event_date",
    "call_date", "call_received_at", "incident_date", "observed_at", "requested_at", "opened_at",
    "sent_at", "published_at", "submitted_at", "placed_at", "dispatched_at", "measured_at",
    "timestamp", "datetime", "date", "time", "ts", "created_at",
)


def _named_event_clock(columns):
    names = {str(column).lower(): column for column in columns}
    return next((names[name] for name in _EVENT_CLOCKS if name in names), None)


def _task_clock(task_text):
    match = re.search(r"(?:time column|event time column|колонка времени|поле времени|время события)"
                      r"\s*[:=—-]\s*[\x60'\"]?(\w+)", task_text, re.I)
    return match[1] if match else None


def detect_time_column(X_train, X_test, time_column=None):
    common = X_train.columns.intersection(X_test.columns)
    if time_column is not None:
        if time_column not in common:
            raise ValueError(f"Поле времени {time_column} отсутствует в train или test.")
        return time_column
    clock = _named_event_clock(common)
    if clock is not None:
        if str(clock).lower() == "time":
            sample = X_train[clock].dropna().head(256).astype(str)
            if sample.str.contains(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}").any():
                return clock
        else:
            return clock
    for col in common:
        if re.fullmatch(r"(transaction|order|measurement|record|observation|pickup|dispatch)"
                        r"_(date|datetime|timestamp|time|ts)", str(col), re.I):
            return col
    for col in common:
        if str(col).lower() == "year":
            values = pd.to_numeric(X_train[col].dropna(), errors="coerce")
            if len(values) and values.between(1900, 2200).all() and values.nunique() >= 2:
                return col
        if str(col).lower() in {"period", "year_factor", "year_index"}:
            past = pd.to_numeric(X_train[col], errors="coerce")
            future = pd.to_numeric(X_test[col], errors="coerce")
            if (past.nunique() >= 2 and len(future) and math.isfinite(past.min())
                    and math.isfinite(future.max()) and past.notna().all() and future.notna().all()
                    and past.max() < future.min()):
                return col
    # Private schemas sometimes use an opaque clock name. A unique date column
    # whose test interval is entirely later supplies evidence of a forecast task.
    candidates = []
    for col in common:
        if re.search(r"birth|profile|registr|signup|expir|deadline|renewal|dob|due_date", str(col), re.I):
            continue
        left, right = X_train[col].dropna(), X_test[col].dropna()
        if left.empty or right.empty or pd.api.types.is_numeric_dtype(left):
            continue
        if not pd.api.types.is_datetime64_any_dtype(left):
            probe = left.iloc[:32].astype(str)
            if not probe.str.contains(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}").all():
                continue
        past = to_utc_naive(left)
        future = to_utc_naive(right)
        if past.notna().all() and future.notna().all() and past.nunique() > 1 and past.max() < future.min():
            candidates.append(col)
    return candidates[0] if len(candidates) == 1 else None


def detect_split(X_train, X_test, max_groups=50, min_unseen=0.9, time_column=None, group_column=None):
    """Определяет временную ось событий или колонку независимых групп.

    Приоритет у времени события и числовых периодов, когда test следует за train.
    Для остальных колонок проверяем долю новых групп в test. Возвращает словарь
    с типом time/group и именем колонки или None для обычной валидации.
    """
    if group_column is not None and time_column is None:
        if group_column not in X_train or group_column not in X_test:
            raise ValueError(f"Group column {group_column} must be present in train and test")
        return {"type": "group", "column": group_column}
    clock = detect_time_column(X_train, X_test, time_column)
    if clock is not None:
        return {"type": "time", "column": clock}
    best, best_share = None, 0.0
    for name in X_test.columns:
        if name not in X_train:
            continue
        n_groups = X_train[name].nunique()
        if n_groups < 2:
            continue
        if n_groups > max_groups and n_groups * 2 > X_train[name].notna().sum():
            continue
        values = X_test[name].dropna()
        if values.empty:
            continue
        share = float((~values.isin(set(X_train[name].dropna()))).mean())
        if share >= min_unseen and share > best_share:
            best, best_share = name, share
    return {"type": "group", "column": best} if best else None


def feature_coverage(X_train, X_test, clock=None):
    """Describe feature availability without storing raw categories, IDs or labels."""
    report = {"train_rows": len(X_train), "test_rows": len(X_test), "features": {}}
    for col in X_train:
        if col not in X_test:
            continue
        past, future = X_train[col], X_test[col]
        field = {"train_missing": int(past.isna().sum()), "test_missing": int(future.isna().sum())}
        field.update(train_unique=int(past.nunique()), test_unique=int(future.nunique()))
        if col == clock:
            bounds = {}
            for part, values in (("train", past), ("test", future)):
                times = (values if pd.api.types.is_numeric_dtype(values) else
                         parse_event_datetime(values))
                bounds[part] = {"min": str(times.min()) if times.notna().any() else None,
                                "max": str(times.max()) if times.notna().any() else None,
                                "invalid_or_missing": int(times.isna().sum())}
            field["time_bounds"] = bounds
        elif (not pd.api.types.is_numeric_dtype(past) or pd.api.types.is_bool_dtype(past)
              or col.lower().endswith(("_id", "_code"))
              or (pd.api.types.is_integer_dtype(past) and field["train_unique"] <= 32)):
            counts = past.value_counts(dropna=True)
            counts = counts[counts > 0]
            known = future.notna() & future.isin(counts.index)
            field.update(train_most_frequent_rows=int(counts.max()) if len(counts) else 0,
                         test_known_rows=int(known.sum()),
                         test_unseen_rows=int((future.notna() & ~known).sum()))
        report["features"][col] = field
    return report


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
    if split and split["type"] == "time":
        warnings.append(f"Временная валидация по колонке '{split['column']}'.")
    elif split:
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
    ("pr_auc", r"pr[\s_-]*auc|average[\s_-]*precision|\bap\b"),
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
    target = targets[0]
    if len(targets) > 1:
        declared = [c for c in targets if c in dictionary.loc[
            dictionary["role"].str.lower().isin(["label", "target"]), "feature"].values]
        named = [c for c in targets if re.search(
            rf"(?:target|label|целевая (?:переменная|колонка)|таргет)(?: column)?[ :=*\x60'\"-]*"
            rf"{re.escape(c)}(?!\w)", task_text, re.I)]
        in_template = [c for c in targets if c in sample]
        for candidates in (named, declared, in_template):
            if len(candidates) == 1:
                target = candidates[0]
                confident = True
                break
        else:
            raise ValueError("Несколько возможных таргетов: " + ", ".join(targets))
    matching_ids = [c for c in ids if sample[c].is_unique and
                    set(sample[c]) == set(test[c])]
    if len(matching_ids) != 1:
        raise ValueError("Не удалось однозначно сопоставить ID test и sample_submission.")
    identifier = matching_ids[0]
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
        single_binary = y.nunique() == 1 and len(sample.columns) == 2
        task_type = "binary" if y.nunique() == 2 or single_binary else "multiclass"
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
    if task_type == "binary" and spec["probabilities"] and len(y) and y.isin([0, 1]).all():
        spec["positive_class"] = 1
    explicit_clock = _task_clock(task_text)
    if explicit_clock:
        spec["time_column"] = detect_time_column(train, test, explicit_clock)
    else:
        group = re.search(r"(?:group(?:ing)? column|group key|колонка группировки|поле группы)"
                          r"\s*[:=—-]\s*[`'\"]?(\w+)", task_text, re.I)
        if group and group[1] in train and group[1] in test:
            spec["group_column"] = group[1]
    spec.update(target_constraints(task_text, target, y, task_type))
    return spec, confident


def target_constraints(text, target, y, task_type):
    if task_type != "regression":
        return {}
    observed = y.dropna()
    numeric = pd.to_numeric(observed, errors="coerce").dropna()
    spec = {"nonnegative": bool(re.search(r"non[- ]?negative|неотрицательн", text, re.I)
                                or (len(numeric) and numeric.min() >= 0))}
    unit = re.search(r"label delay unit\s*:\s*`?(minutes?|hours?)", text, re.I)
    duration = re.search(r"(?:минут\w*|час(?:а|ов)?|minutes?|hours?)\s+(?:от|до|между|until|from|between|to)\b"
                         r"|(?:time|duration|delay)\s+(?:elapsed|until|between|from|to)\b", text, re.I)
    delay_unit = None
    if len(numeric) != len(observed) or not len(numeric) or not np.isfinite(numeric).all() or numeric.min() < 0:
        return spec
    if unit and (duration or re.search(r"label.*(?:available|observed).*after|метк.*(?:доступн|известн).*после", text, re.I)):
        delay_unit = "hours" if unit[1].lower().startswith("hour") else "minutes"
    elif duration:
        part = text[max(0, duration.start()-40):duration.end()+100] + " " + target
        if re.search(r"минут|minute", part, re.I):
            delay_unit = "minutes"
        elif re.search(r"час|hour", part, re.I):
            delay_unit = "hours"
    if delay_unit and 0 < numeric.median() <= (30 * 24 * (60 if delay_unit == "minutes" else 1)):
        spec["label_delay_unit"] = delay_unit
    return spec


_START_NAMES = {"hour_start", "hour_start_time", "hour_start_ts", "hour_start_at", "interval_start",
                "interval_start_time", "period_start", "bucket_start"}
_END_NAMES = {"hour_end", "hour_end_time", "hour_end_ts", "hour_end_at", "interval_end",
              "interval_end_time", "period_end", "bucket_end"}
_EVENT_CHUNK_ROWS = 20_000
_AUX_ROW_LIMIT = 1_000_000
_AUX_MEMORY_LIMIT = 192 * 1024**2


_GENERIC_HOUR_COLUMNS = {"timestamp", "datetime", "hour", "hour_ts", "hour_timestamp", "ts", "hourly_timestamp"}
_VERSION_CLOCKS = ("updated_at", "modified_at", "valid_from", "created_at", "updated_time", "creation_time")
_PUBLICATION_CLOCKS = ("available_at", "published_at", "released_at", "publication_time", "publish_time", "ingested_at")
_VERSION_FIELDS = {"version", "revision", "version_id", "revision_id", "revised_at",
                   "valid_to", "is_current", "snapshot_date", "snapshot_at"}


def _dictionary_clocks(conn, tables, train, test, deadline, log):
    """Read bounded, explicitly attributed clock descriptions in the current snapshot."""
    import time
    from postgres_loader import _identifier, _relation, _set_timeout

    candidates = [t for t in tables.values() if t not in (train, test) and t.name.lower() != "data_dictionary"
                  and any(c.lower() in _GENERIC_HOUR_COLUMNS or _temporal_field(t, c) for c in t.columns)]
    if not candidates:
        return {}
    names = sorted({f"{t.schema}.{t.name}" for t in candidates} | {t.name for t in candidates})
    clock_names = sorted({c for t in candidates for c in t.columns
                          if c.lower() in _GENERIC_HOUR_COLUMNS or _temporal_field(t, c)})
    descriptions = {}
    for table in tables.values():
        if table.name.lower() != "data_dictionary":
            continue
        aliases = ({"table_name", "table", "source_table", "relation_name"},
                   {"column_name", "column", "field_name", "field", "feature_name", "feature"},
                   {"description", "column_description", "definition", "comment", "field_description"},
                   {"table_schema", "schema_name", "schema"})
        fields = [[c for c in table.columns if c.lower() in variants] for variants in aliases]
        qualified = [c for c in table.columns if c.lower() in {"qualified_column_name", "column_path", "feature_path"}]
        if not qualified and not fields[0] and len(fields[1]) == 1 and fields[1][0].lower() in {"feature", "feature_name"}:
            qualified = fields[1]
        if len(fields[2]) != 1 or len(fields[3]) > 1:
            log(f"Метаданные {table.schema}.{table.name}: неоднозначная схема описаний")
            continue
        description = _identifier(fields[2][0])
        schema = _identifier(fields[3][0]) + "::text" if fields[3] else "NULL::text"
        if len(fields[0]) == len(fields[1]) == 1:
            relation, column = [_identifier(group[0]) for group in fields[:2]]
            projection = f"{relation}::text, {column}::text"
            where = f"{relation}::text = ANY(%s) AND {column}::text = ANY(%s)"
            params = (names, clock_names)
        elif len(qualified) == 1 and not fields[0]:
            # Match only complete table.column or schema.table.column paths, never a bare timestamp.
            column = _identifier(qualified[0])
            projection = f"NULL::text, {column}::text"
            where = f"{column}::text = ANY(%s)"
            params = ([f"{relation}.{clock}" for relation in names for clock in clock_names],)
        else:
            log(f"Метаданные {table.schema}.{table.name}: неоднозначная схема описаний")
            continue
        sql = (f"SELECT {projection}, left({description}::text, 1025), {schema} "
               f"FROM {_relation(table)} WHERE {where} LIMIT 1001")
        try:
            with conn.transaction():
                stop = time.monotonic() + 5
                _set_timeout(conn, min(deadline, stop) if deadline is not None else stop)
                rows = conn.execute(sql, params).fetchall()
        except Exception as exc:
            log(f"Метаданные {table.schema}.{table.name}: чтение пропущено ({type(exc).__name__})")
            continue
        if len(rows) > 1000:
            log(f"Метаданные {table.schema}.{table.name}: превышен лимит описаний")
            continue
        scoped_count = 0
        for relation_name, col, text, schema_name in rows:
            if relation_name is None and isinstance(col, str):
                relation_name, _, col = col.rpartition(".")
            matches = [t for t in candidates if relation_name in (t.name, f"{t.schema}.{t.name}")
                       and (schema_name is None or schema_name == t.schema)]
            if len(matches) != 1 or col not in matches[0].columns:
                continue
            t = matches[0]
            entries = descriptions.setdefault((t.schema, t.name), {})
            # Only one complete description can establish a clock for a given field.
            entries[col] = text if col not in entries and text and len(text) <= 1024 else None
            scoped_count += 1
        log(f"Метаданные {table.schema}.{table.name}: привязано строк описаний времени {scoped_count}")
    return descriptions


def _hour_clock(table, task_text, descriptions=None):
    """Resolve timestamp semantics only from field names or scoped descriptions."""
    clocks = {}
    ambiguous_clocks = set()
    for column in table.columns:
        name = column.lower()
        if name in _VERSION_CLOCKS or name in _PUBLICATION_CLOCKS:
            continue
        if name not in _START_NAMES | _END_NAMES | _GENERIC_HOUR_COLUMNS and not _temporal_field(table, column):
            continue
        if name in _START_NAMES:
            # An arbitrary interval can be longer than an hour on a private schema.
            if "hour" in name:
                clocks[column] = "start"
                continue
        if name in _END_NAMES:
            clocks[column] = "end"
            continue
        # Accept a generic clock only with unambiguous documentation for this exact field.
        scoped = [line for line in task_text.splitlines()
                  if re.search(rf"(?<!\w){re.escape(table.name)}\s*\.\s*{re.escape(column)}(?!\w)",
                               line.replace(chr(96), ""), re.I)]
        dictionary = descriptions or {}
        if column in dictionary and dictionary[column] is None:
            ambiguous_clocks.add(column)
            continue
        evidence = set()
        phrases = {"start": r"start of (?:the |an? )?hour|hour start|начал[оуа] (?:каждого )?часа",
                   "end": r"end of (?:the |an? )?hour|hour end|(?:конец|конца|окончани[ея]) часа"}
        prefix = rf"(?:^\s*|(?<!\w)(?:{re.escape(column)}|timestamp)(?:\s+(?:is|marks|denotes|means|"
        prefix += r"соответствует|означает|обозначает))?\s*[:=—-]?\s*(?:the )?)"
        for fragment in [table.comments.get(column) or "", dictionary.get(column) or "", *scoped]:
            fragment = fragment.replace(chr(96), "")
            ambiguous = re.search(r"\b(?:not|unknown|unclear|maybe|might|не|нет|неясно|неизвестно|возможно)\b|\?",
                                  fragment, re.I)
            mentioned = [mode for mode, pattern in phrases.items() if re.search(pattern, fragment, re.I)]
            if ambiguous or len(mentioned) > 1:
                evidence.add("uncertain")
            elif mentioned and re.search(prefix + "(?:" + phrases[mentioned[0]] + r")\b", fragment, re.I):
                evidence.add(mentioned[0])
        if len(evidence) == 1 and "uncertain" not in evidence:
            clocks[column] = evidence.pop()
        elif evidence or column in dictionary or table.comments.get(column) or scoped:
            ambiguous_clocks.add(column)
    # Prefer the published interval end when both bounds are present.
    for mode in ("end", "start"):
        matches = [col for col, value in clocks.items() if value == mode]
        if len(matches) == 1:
            return matches[0], mode
        if matches:
            break
    timestamps = [c for c in table.columns
                  if ("timestamp" in table.column_types.get(c, "") or table.column_types.get(c) == "date"
                      or c.lower() in _START_NAMES | _GENERIC_HOUR_COLUMNS)
                  and c.lower() not in _PUBLICATION_CLOCKS]
    if len(timestamps) == 1 and timestamps[0].lower() not in _VERSION_CLOCKS and timestamps[0] not in ambiguous_clocks:
        return timestamps[0], "inferred"
    return None, None


def _publication_clock(table, task_text):
    """A plain timestamp or last-update clock cannot reconstruct first-published payloads."""
    names = {c.lower(): c for c in table.columns}
    if names.keys() & {"version", "revision", "version_id", "revision_id", "revised_at", "updated_at",
                       "modified_at", "valid_from", "valid_to", "is_current", "snapshot_date", "snapshot_at"}:
        raise ValueError("не подтверждена история публикации и исправлений измерений")
    publication = [names[k] for k in _PUBLICATION_CLOCKS if k in names]
    if not publication:
        return None
    if len(publication) == 1:
        col = publication[0]
        full = f"{table.schema}.{table.name}.{col}"
        declared = re.search(rf"^\s*Immutable observation publication clock\s*:\s*{re.escape(full)}\s*$",
                             task_text.replace(chr(96), ""), re.M | re.I)
        comment = (table.comments.get(col) or "").strip()
        uncertain = re.search(r"\b(?:not|may|last|latest|updated|unknown|revised|не|возможно)\b|\?", comment, re.I)
        documented = (re.search(r"\bimmutable\b|неизмен", comment, re.I) and
                      re.search(r"\bfirst (?:publication|release|availability)\b|\bfirst (?:published|available)\b"
                                r"|перв\w* публикаци", comment, re.I))
        if not uncertain and (declared or documented):
            return col
    raise ValueError("не подтверждено время первой публикации неизменяемого измерения")


class _NoEventRelation(ValueError):
    """A direct link is missing; a validated dimension may later supply it."""


def _bridge_routes(table, parents):
    """Single-hop paths through already installed, direct static dimensions."""
    routes = {}
    for parent, fields in parents:
        declared = [fk for fk in parent.foreign_keys if tuple(fk.target) == (table.schema, table.name)]
        links = {(fk.columns[0], fk.target_columns[0]) for fk in declared
                 if len(fk.columns) == len(fk.target_columns) == 1}
        links.update((fk.target_columns[0], fk.columns[0]) for fk in table.foreign_keys
                     if tuple(fk.target) == (parent.schema, parent.name)
                     and len(fk.columns) == len(fk.target_columns) == 1)
        if not declared and not links and table.schema == parent.schema:
            candidates = [c for c in fields if c in table.columns and c.lower().endswith(("_id", "_key", "_code"))]
            key = (table.primary_key[0] if len(table.primary_key) == 1 else None) or (
                candidates[0] if len(candidates) == 1 else None)
            if key and key in table.columns:
                links.add((key, key))
            hour_col, _ = _hour_clock(table, "")
            if hour_col in table.primary_key and len(table.primary_key) == 2:
                entity = next(c for c in table.primary_key if c != hour_col)
                if entity in candidates:
                    links.add((entity, entity))
        for via, right in links:
            if via in fields and right in table.columns:
                routes[(parent.name + "__" + via, right)] = parent, via
    return routes


def _temporal_field(table, col):
    name = col.lower()
    return ("timestamp" in table.column_types.get(col, "") or table.column_types.get(col) == "date"
            or name in {"valid_from", "valid_to", "as_of", "date", "time"}
            or name.endswith(("_date", "_at", "_timestamp", "_time")))


def _plan_relation(table, train, test, identifier, train_only, task_text, descriptions=None, bridges=None):
    from leak_audit import restricted_inputs
    common = set(train.columns).intersection(test.columns)
    name = table.name.lower()
    if (table in (train, test) or "submission" in name or
            re.search(r"(?:^|_)(?:labels?|targets?|outcomes?|train|test)(?:_|$)", name)):
        return None
    if name == "data_dictionary":
        raise ValueError("словарь метаданных")
    restrictions = restricted_inputs(task_text, [table.name, f"{table.schema}.{table.name}",
                                                 *[f"{table.name}.{c}" for c in table.columns]])
    if table.name in restrictions or f"{table.schema}.{table.name}" in restrictions:
        raise ValueError("task.md does not establish safe availability for the source")
    hour_col, mode = _hour_clock(table, task_text, descriptions)
    names = {c.lower() for c in table.columns}
    dated_columns = [c for c in table.columns if _temporal_field(table, c) or c.lower() in _GENERIC_HOUR_COLUMNS]
    hourly = hour_col is not None
    if not hourly and (names & (_START_NAMES | _END_NAMES | set(_PUBLICATION_CLOCKS)) or
                       (len(dated_columns) < 2 and names & _GENERIC_HOUR_COLUMNS)):
        raise ValueError("не подтверждены границы интервала или однозначная временная колонка")
    if (not hourly or mode == "inferred") and re.search(r"(?:^|_)events?(?:_|$)", name):
        raise ValueError("таблица событий требует отдельной связи и правила доступности, статический JOIN недопустим")
    publication = _publication_clock(table, task_text) if hourly else None
    typed_clocks = [c for c in common if "timestamp" in train.column_types.get(c, "")
                    and "timestamp" in test.column_types.get(c, "")]
    event_clock = _task_clock(task_text) or _named_event_clock(common) or (typed_clocks[0] if len(typed_clocks) == 1 else None)
    event_col = event_clock if hourly else None
    if hourly and event_col not in common:
        raise ValueError("не подтверждено время события")

    relations = {(fk.columns[0], fk.target_columns[0])
                 for main in (train, test) for fk in main.foreign_keys
                 if tuple(fk.target) == (table.schema, table.name) and len(fk.columns) == 1
                 and len(fk.target_columns) == 1 and fk.columns[0] in common}
    if not relations and train.schema == test.schema == table.schema:
        keys = [c for c in table.columns if c in common and c != identifier
                and (c.lower().endswith(("_id", "_key", "_code")) or c in table.primary_key) and c != hour_col]
        if not keys and mode == "inferred":
            keys = [c for c in table.columns if c in common and c not in {identifier, hour_col, event_clock}
                    and c not in train_only and not _temporal_field(table, c)]
        key = table.primary_key[0] if len(table.primary_key) == 1 and table.primary_key[0] in common else None
        key = key or (keys[0] if len(keys) == 1 else None)
        if key and key in common and key in table.columns:
            relations.add((key, key))
    if not relations and bridges:
        relations = set(bridges)
    if not relations:
        raise _NoEventRelation("нет однозначной связи с событиями")
    if len(relations) != 1:
        raise ValueError("неоднозначный путь к справочнику")
    left, right = relations.pop()
    if left == identifier or left in train_only:
        raise ValueError("ключ совпадает с идентификатором события или результатом")
    # Versioned entity snapshots use only strictly past profiles.
    if not hourly:
        timeline = next((c for name in _VERSION_CLOCKS for c in table.columns if c.lower() == name), None)
        event_col = event_clock
        if names & ({"valid_to"} if timeline else _VERSION_FIELDS):
            raise ValueError("не подтверждён интервал действительности версии")
        if timeline and event_col not in common:
            raise ValueError("не подтверждено время события для версии справочника")
        if timeline and event_col in common:
            hour_col, mode = timeline, "profile"
        else:
            event_col = None
            if len(dated_columns) >= 2:
                mode = "static_dates"
    blocked = {x.lower() for x in train_only} | {identifier.lower(), "label", "outcome", "prediction", "predicted",
                  "submission", "clicks", "ctr", "conversions", "impressions", "engagement", "future", "next",
                  "cumulative", "total", "target"}
    unsupported_types = {"bytea", "json", "jsonb", "array", "user-defined"}
    values = []
    for c in table.columns:
        low = c.lower()
        if (c == right or c == hour_col or c == publication or low in _START_NAMES | _END_NAMES or
                c in dated_columns or f"{table.name}.{c}" in restrictions or
                any(re.search(rf"(?:^|_){re.escape(term)}(?:_|$)", low) for term in blocked)
                or table.column_types.get(c, "").lower() in unsupported_types):
            continue
        if not hourly:
            if re.search(r"(?:^|_)(?:count|sum|last|score)(?:_|$)", low):
                continue
            if mode == "profile" and hour_col.lower() in {"created_at", "creation_time"}:
                immutable = re.search(r"\b(?:immutable|fixed at creation)\b|неизменяем", table.comments.get(c, ""), re.I)
                stable = re.search(r"(?:^|_)(?:gender|sex|birth_year|category|type)(?:_|$)", low)
                if not (immutable or stable):
                    continue
        values.append(c)
    if not values:
        raise ValueError("нет доступных признаков")
    return left, right, event_col, hour_col, mode, values


def label_available_at(times, y_fit, unit):
    import numpy as np

    if unit not in ("minutes", "hours") or not pd.api.types.is_datetime64_any_dtype(times):
        raise ValueError("A duration label needs a datetime event clock and unit 'minutes'")
    duration = np.asarray(y_fit, dtype=float)
    if np.any(duration < 0) or np.isinf(duration).any():
        raise ValueError("Label delays must be nonnegative and finite")
    return to_utc_naive(times) + pd.to_timedelta(duration, unit={"minutes":"m","hours":"h"}[unit])


def to_utc_naive(series):
    """Reject clock-only strings: a calendar date filled in at run time is not event evidence."""
    if not pd.api.types.is_datetime64_any_dtype(series):
        text = series.astype("string").str.strip()
        clock_only = text.str.fullmatch(r"\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?\s*(?:AM|PM)?\s*(?:Z|[+-]\d{2}:?\d{2})?",
                                       flags=re.I, na=False)
        series = series.mask(clock_only)
    return pd.to_datetime(series, errors="coerce", utc=True, format="mixed").dt.tz_convert(None).astype("datetime64[ns]")


parse_event_datetime = to_utc_naive


def _infer_interval(entities, stamps, before, *, diagnostics=None):
    """Establish cadence from distinct historical timestamps within each entity."""
    if before is None or pd.isna(before):
        raise ValueError("неопределимый шаг: нет времени первого события")
    past = pd.DataFrame({"key": entities, "stamp": stamps})
    past = past.loc[past["stamp"].lt(before)]
    unique = past.drop_duplicates()
    if diagnostics is not None:
        diagnostics["duplicate_timestamps"] = len(past) - len(unique)
    past = unique.sort_values(["key", "stamp"])
    gaps = past.groupby("key", sort=False)["stamp"].diff().dropna()
    if len(gaps) < 2:
        raise ValueError("неопределимый шаг: нужно не менее двух исторических интервалов")
    step = gaps.median()
    share = gaps.eq(step).mean()
    if step <= pd.Timedelta(0) or share < .8 or not (gaps >= step).all() or (gaps % step != pd.Timedelta(0)).any():
        raise ValueError(f"irregular_step share={share:.3f} median={step} (нерегулярный шаг)")
    return step


def _reference_mode(table, auxiliary, right, clock, mode, before, publication, values):
    if mode == "static_dates":
        if auxiliary[right].duplicated().any():
            raise ValueError("non_unique_key_with_multiple_dates")
        return None, "static", "one_row_per_key_multiple_dates", values
    if mode != "inferred":
        return clock, mode, "documented_clock" if clock else "keyed_dimension", values
    stamps = to_utc_naive(auxiliary[clock])
    historical = auxiliary.loc[stamps.lt(before), [right]].assign(stamp=stamps[stamps.lt(before)])
    counts = historical.drop_duplicates().groupby(right, dropna=True).size()
    if counts.median() >= 3:
        return clock, mode, "regular_step", values
    names = {c.lower() for c in table.columns}
    # Explicit measurement hints cannot be treated as timeless profile attributes.
    if (publication or clock.lower() in _START_NAMES | _GENERIC_HOUR_COLUMNS | {"day", "date"}
            or re.search(r"^(?:measurement|measured|observation|observed|recorded|sampled)_", clock, re.I)):
        raise ValueError("undetermined_step (неопределимый шаг): median historical timestamps per key < 3")
    values = [c for c in values if not re.search(r"(?:^|_)(?:count|sum|last|score)(?:_|$)", c, re.I)]
    if not values:
        raise ValueError("нет доступных признаков справочника")
    if not auxiliary[right].duplicated().any():
        return clock, "dated_static", "one_row_per_key", values
    # A version is safe only when the column denotes when that version was created.
    if re.search(r"^(?:created|registered|registration|signup|signed_up|creation)_", clock, re.I):
        return clock, "dated_profile", "version_creation_clock", values
    if names.intersection(_VERSION_FIELDS):
        raise ValueError("ambiguous_version_clock")
    raise ValueError("undetermined_step: ambiguous timestamps with fewer than three historical observations per key")


def _prepare_lookup(auxiliary, right, hour_col, mode, values, prefix, publication=None, *, coalesce=False,
                    cadence_before=None, diagnostics=None):
    """Validate keys after normalisation, before any train/test rows are touched."""
    auxiliary = auxiliary.reset_index(drop=True)
    lookup = auxiliary[values].add_prefix(prefix).copy()
    payload = list(lookup)
    lookup["__entity"] = auxiliary[right].astype("string")
    lookup = lookup.loc[lookup["__entity"].notna() & lookup["__entity"].str.strip().ne("")].copy()
    if not hour_col:
        repeated = lookup["__entity"].duplicated(keep=False)
        if repeated.any():
            # A current snapshot cannot tell which conflicting version was visible.
            dup = lookup.loc[repeated]
            for column in payload:
                unique = dup.groupby("__entity", sort=False)[column].transform("nunique", dropna=False)
                lookup.loc[unique.index[unique.gt(1)], column] = None
        return lookup.drop_duplicates("__entity", keep="last").set_index("__entity")
    stamps = to_utc_naive(auxiliary.loc[lookup.index, hour_col])
    if mode in {"profile", "dated_static", "dated_profile"}:
        lookup["__available"] = stamps
        lookup = lookup.dropna(subset=["__available"]).sort_values("__available", kind="stable")
        if mode == "profile":
            lookup[prefix + "profile_count"] = lookup.groupby("__entity", sort=False).cumcount() + 1
        elif mode == "dated_profile":
            repeats = lookup.duplicated(["__entity", "__available"], keep=False)
            for column in payload:
                conflicts = lookup.loc[repeats].groupby(["__entity", "__available"])[column].transform("nunique", dropna=False)
                lookup.loc[conflicts.index[conflicts.gt(1)], column] = None
        return lookup.drop_duplicates(["__entity", "__available"], keep="last")
    if mode not in {"start", "end", "inferred"}:
        raise ValueError("Unknown interval semantics")
    hour = pd.Timedelta(hours=1)
    step = _infer_interval(lookup["__entity"], stamps, cadence_before, diagnostics=diagnostics) if mode == "inferred" else hour
    if mode == "inferred":
        lookup["__source_stamp"] = stamps
    lookup["__interval_end"] = stamps + step if mode in {"start", "inferred"} else stamps
    lookup["__available"] = lookup["__interval_end"]
    if publication:
        release = to_utc_naive(auxiliary.loc[lookup.index, publication])
        lookup["__available"] = lookup["__available"].where(lookup["__available"] >= release, release)
    lookup = lookup.dropna(subset=["__available", "__interval_end"])
    keys = ["__entity", "__interval_end"]
    if coalesce:
        lookup = lookup.drop_duplicates()
        repeated = lookup.duplicated(keys, keep=False)
        if repeated.any():
            if publication:
                raise ValueError("conflicting revisions with publication clocks require version history")
            lookup.loc[repeated, payload] = None
            lookup = lookup.sort_values("__available").drop_duplicates(keys, keep="last")
    if lookup.duplicated(keys).any():
        raise ValueError("неуникальный почасовой ключ")
    numeric = {}
    for c in payload:
        values = pd.to_numeric(lookup[c], errors="coerce")
        if values.notna().any() and (lookup[c].isna() | values.notna()).all():
            numeric[c] = values.astype("float32")
    grid = pd.DataFrame(numeric, index=lookup.index)
    grid["__available"] = lookup["__available"]
    if mode == "inferred":
        grid["__source_stamp"] = lookup["__source_stamp"]
    grid.index = pd.MultiIndex.from_frame(lookup[keys])
    lookup = lookup.sort_values(["__available", "__interval_end"], kind="stable")
    latest = lookup.groupby("__entity", sort=False)["__interval_end"].cummax()
    lookup = lookup.loc[lookup["__interval_end"].eq(latest)].reset_index(drop=True)
    if step == hour:
        lookup.attrs["grid"] = grid
    if mode == "inferred":
        lookup.attrs["inferred_step"] = step
    return lookup


def _hour_features(grid, entities, events):
    """Exact completed-hour lags; a missing or unpublished bucket contributes no value."""
    outputs = {}
    total = count = None
    floor = events.dt.floor("h")
    latest_end = floor.where(floor.ne(events), floor - pd.Timedelta(hours=1))
    delayed = (grid["__available"].to_numpy() != grid.index.get_level_values(1).to_numpy()).any()
    for offset in range(168 if delayed else 6):
        end = latest_end - pd.Timedelta(hours=offset)
        block = grid.reindex(pd.MultiIndex.from_arrays([entities, end])).reset_index(drop=True)
        block = block.drop(columns="__source_stamp", errors="ignore")
        available = block.pop("__available").lt(events.reset_index(drop=True))
        values = block.where(available, axis=0).to_numpy(dtype="float64")
        if total is None:
            total = np.zeros_like(values)
            count = np.zeros(values.shape, dtype=np.uint16)
        finite = np.isfinite(values)
        total += np.where(finite, values, 0)
        count += finite
        if offset < 3:
            outputs.update({f"{col}__lag{offset+1}h": values[:, j] for j, col in enumerate(block)})
        if offset in (2, 5, 23, 167):
            means = np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)
            outputs.update({f"{col}__mean{offset+1}h": means[:, j] for j, col in enumerate(block)})
    if not delayed:
        outputs.update(_hour_rollups(grid, entities, latest_end))
    for col in grid.columns.difference(["__available", "__source_stamp"], sort=False):
        outputs[f"{col}__delta1_3h"] = outputs[f"{col}__lag1h"] - outputs[f"{col}__lag3h"]
        outputs[f"{col}__dev7d"] = outputs[f"{col}__lag1h"] - outputs[f"{col}__mean168h"]
    return pd.DataFrame(outputs)


def _hour_rollups(grid, entities, latest_end):
    cols = grid.columns.difference(["__available", "__source_stamp"], sort=False)
    windows = {24: np.full((len(entities), len(cols)), np.nan), 168: np.full((len(entities), len(cols)), np.nan)}
    for key, positions in pd.Series(np.arange(len(entities))).groupby(entities.reset_index(drop=True)):
        try:
            rows = grid.xs(key).sort_index()
        except KeyError:
            continue
        rows = rows.loc[rows.index == rows.index.floor("h"), cols]
        stamps = rows.index.to_numpy()
        values = rows.to_numpy(dtype="float64")
        finite = np.isfinite(values)
        sums = np.vstack([np.zeros(len(cols)), np.where(finite, values, 0).cumsum(axis=0)])
        counts = np.vstack([np.zeros(len(cols)), finite.cumsum(axis=0)])
        idx = positions.to_numpy()
        end = latest_end.iloc[idx].to_numpy()
        right = np.searchsorted(stamps, end, side="right")
        for width, target in windows.items():
            left = np.searchsorted(stamps, end - np.timedelta64(width - 1, "h"))
            n = counts[right] - counts[left]
            target[idx] = np.divide(sums[right] - sums[left], n, out=np.full(n.shape, np.nan),
                                    where=(n > 0) & pd.notna(end)[:, None])
    return {f"{col}__mean{width}h": table[:, i] for width, table in windows.items() for i, col in enumerate(cols)}


def _join_chunk(events, lookup, left, event_col, audit=None):
    payload = [c for c in lookup if c not in ("__entity", "__available", "__interval_end", "__source_stamp")]
    if set(events).intersection(payload):
        raise ValueError("совпадающие имена признаков")
    keys = events[left].astype("string")
    if event_col:
        times = to_utc_naive(events[event_col])
        query = pd.DataFrame({"__entity": keys.array, "__event": times.array})
        query["__pos"] = query.index
        query = query.dropna(subset=["__event", "__entity"]).sort_values("__event", kind="stable")
        aligned = pd.merge_asof(query, lookup, by="__entity", left_on="__event", right_on="__available",
                                direction="backward", allow_exact_matches=False)
        matched = aligned["__available"].notna()
        if not (aligned.loc[matched, "__available"] < aligned.loc[matched, "__event"]).all():
            raise ValueError("Интервал измерений не закончился до события")
        if audit is not None:
            lag = (aligned.loc[matched, "__event"] - aligned.loc[matched, "__available"]).dt.total_seconds()
            if "__interval_end" in aligned and matched.any():
                age = (aligned.loc[matched, "__event"] - aligned.loc[matched, "__interval_end"]).dt.total_seconds()
                audit["min_interval_age_seconds"] = min(float(age.min()), audit.get("min_interval_age_seconds", float("inf")))
                audit["max_interval_age_seconds"] = max(float(age.max()), audit.get("max_interval_age_seconds", 0))
            if len(lag):
                for key, op in (("min_lag_seconds", min), ("max_lag_seconds", max)):
                    value = float(op(lag))
                    audit[key] = op(value, audit.get(key, value))
            audit["invalid_event_time_rows"] = audit.get("invalid_event_time_rows", 0) + int(times.isna().sum())
        aligned = aligned.set_index("__pos")[payload].reindex(range(len(events)))
        if "grid" in lookup.attrs:
            aligned = pd.concat([aligned, _hour_features(lookup.attrs["grid"], keys, times)], axis=1)
    else:
        aligned = lookup.reindex(keys)[payload]
        if audit is not None:
            matched = keys.isin(lookup.index)
    if audit is not None:
        audit["rows"] += len(events)
        audit["matched_rows"] += int(matched.sum())
        audit["missing_key_rows"] += int((keys.isna() | keys.str.strip().eq("")).sum())
        for col in audit.get("feature_nonmissing", ()):
            audit["feature_nonmissing"][col] += int((aligned[col].notna() & aligned[col].ne("")).sum())
    aligned.index = events.index
    return pd.concat([events, aligned], axis=1)


def _audit_lookup(samples, lookup, left, event_col, budget, source_name):
    from leak_audit import sample_positions

    def verify(counts, check_time):
        parts = []
        for (events, features), count in zip(samples, counts):
            ids = sample_positions(len(events), limit=count)
            parts.append((events.iloc[ids], features.iloc[ids]))
        events, expected = (pd.concat(part, ignore_index=True) for part in zip(*parts))
        return _verify_lookup(events, expected, lookup, left, event_col, check_time)

    budget.sampled(source_name, [len(events) for events, _ in samples], verify)


def _verify_lookup(events, expected, lookup, left, event_col, check_time):
    from leak_audit import LeakAuditError, check_equal, replay_batches, replay_payload
    if not event_col:
        return {"features": list(expected), "rule": "static payload; ambiguous versions masked"}
    times = to_utc_naive(events[event_col])
    keys = events[left].astype("string")
    raw = pd.DataFrame(lookup)
    grid = lookup.attrs.get("grid")
    payload = [col for col in raw if not col.startswith("__")]
    # Independent per-event truncation via strict lower bounds on each entity's releases.
    positions = np.full(len(events), -1, dtype=int)
    future = np.full(len(events), -1, dtype=int)
    by_key = raw.groupby("__entity", sort=False).indices
    for key, idx in pd.Series(np.arange(len(events))).groupby(keys, sort=False):
        src = by_key.get(key)
        if src is None:
            continue
        ids = idx.to_numpy()
        stamp = times.iloc[ids].to_numpy()
        at = np.searchsorted(raw.iloc[src]["__available"].to_numpy(), stamp, side="left")
        before = (at > 0) & pd.notna(stamp)
        after = (at < len(src)) & pd.notna(stamp)
        positions[ids[before]] = src[at[before] - 1]
        future[ids[after]] = src[at[after]]
    raw = raw.reset_index(drop=True)
    visible_rows = raw.reindex(positions).reset_index(drop=True)
    step = lookup.attrs.get("inferred_step")
    if step is not None:
        matched = positions >= 0
        actual_ends = visible_rows["__source_stamp"] + step
        if not actual_ends[matched].lt(times.reset_index(drop=True)[matched]).all():
            raise LeakAuditError("inferred interval has not ended: source timestamp + step >= event")
    check_equal(expected[payload], visible_rows[payload])
    # Give every query its own key and noisy suffix. A backward lookup only
    # needs the records immediately to either side of the release boundary.
    withheld = raw.reindex(future).reset_index(drop=True)
    for column in payload:
        values = pd.to_numeric(withheld[column], errors="coerce")
        with_numeric = pd.api.types.is_numeric_dtype(visible_rows[column])
        withheld[column] = (values.fillna(0.) + 987654.321 if with_numeric else "audit future payload")
    synthetic = pd.Series(np.arange(len(events)), dtype="string")
    visible_rows["__entity"] = synthetic
    withheld["__entity"] = synthetic
    changed = pd.concat([visible_rows, withheld], ignore_index=True).dropna(subset=["__available"])
    changed = changed.sort_values("__available", kind="stable")
    audit_key = "__audit_entity"
    query = pd.DataFrame({audit_key: synthetic, event_col: times})
    replay = _join_chunk(query, changed, audit_key, event_col)
    check_equal(expected[payload], replay[payload])
    if grid is not None:
        columns = grid.columns.difference(["__available", "__source_stamp"], sort=False)
        latest = (times - pd.Timedelta(nanoseconds=1)).dt.floor("h").to_numpy()
        values = np.full((168, len(times), len(columns)), np.nan)
        corrupted = np.full_like(values, np.nan)
        for offset in range(-2, 168):
            if offset % 24 == 0:
                check_time()
            index = pd.MultiIndex.from_arrays([keys, latest - np.timedelta64(offset, "h")])
            rows = grid.reindex(index)
            if step is not None and ((rows["__available"].to_numpy() < times.to_numpy()) &
                                     ~((rows["__source_stamp"] + step).to_numpy() < times.to_numpy())).any():
                raise LeakAuditError("inferred grid interval has not ended")
            args = (rows[columns].to_numpy(dtype=float), rows["__available"].to_numpy(), times.to_numpy())
            dirty = replay_payload(*args, corrupt_future=True)
            if offset >= 0:
                values[offset] = replay_payload(*args, corrupt_future=False)
                corrupted[offset] = dirty
            elif np.isfinite(dirty).any():
                raise LeakAuditError("unfinished grid hour was visible")
        for replay in (values, corrupted):
            for i, column in enumerate(columns):
                history = replay[:, :, i]
                check_columns = {f"{column}__lag{lag}h": history[lag-1] for lag in (1, 2, 3)}
                for width in (3, 6, 24, 168):
                    finite = np.isfinite(history[:width])
                    count = finite.sum(axis=0)
                    total = np.where(finite, history[:width], 0).sum(axis=0)
                    check_columns[f"{column}__mean{width}h"] = np.divide(
                        total, count, out=np.full(len(times), np.nan), where=count > 0)
                check_columns[f"{column}__delta1_3h"] = history[0] - history[2]
                check_columns[f"{column}__dev7d"] = history[0] - check_columns[f"{column}__mean168h"]
                check_equal(expected[list(check_columns)], pd.DataFrame(check_columns))
    for idx, cutoff in replay_batches(times, check_time, blocks=2):
        retained = raw["__available"].lt(cutoff)
        past = raw.loc[retained]
        corrupted = raw.copy()
        # Write noise to the payload of every unavailable record, including equality.
        for col in payload:
            corrupted[col] = corrupted[col].astype(object)
            corrupted.loc[~retained, col] = -987654.321
        if grid is not None:
            usable = grid["__available"].lt(cutoff)
            past.attrs["grid"] = grid.loc[usable]
            corrupted_grid = grid.copy()
            corrupted_grid.loc[~usable, corrupted_grid.columns.difference(["__available", "__source_stamp"])] = -987654.
            corrupted.attrs["grid"] = corrupted_grid
        for source in (past, corrupted):
            check_time()
            actual = _join_chunk(events.iloc[idx].reset_index(drop=True), source, left, event_col)
            check_equal(expected.iloc[idx], actual[expected.columns])
    checks = ["per_row_asof", "truncated_source", "corrupted_source", "per_cutoff_corrupted_source"]
    if step is not None:
        checks.append("source_timestamp_plus_inferred_step")
    return dict(features=list(expected), rule="release < event", cutoffs=int(times.nunique()), checks=checks)


def _read_lookup(conn, table, cols, deadline, *, hourly=False):
    import time
    from postgres_loader import _fetch_size, _identifier, _relation, _remaining, _set_timeout

    # A server-side cursor prevents a large dimension from being buffered by libpq.
    fields = ", ".join(map(_identifier, cols))
    order = ""
    if not hourly:
        fields_in_order = list(dict.fromkeys([*cols, *table.primary_key]))
        order = " ORDER BY " + ", ".join(map(_identifier, fields_in_order))
    query = f"SELECT {fields} FROM {_relation(table)}{order} LIMIT {_AUX_ROW_LIMIT + 1}"
    # Hourly district series contain over 500k rows per source. Load one at a time.
    memory_limit = 1024**3 if hourly else _AUX_MEMORY_LIMIT
    _set_timeout(conn, deadline)
    chunks, n_rows, n_bytes = [], 0, 0
    with conn.cursor(name="tyrell_auxiliary") as cursor:
        cursor.execute(query)
        batch = 1024
        last_timeout = time.monotonic()
        while True:
            left = _remaining(deadline)
            now = time.monotonic()
            if left < 120 and now - last_timeout >= min(10.0, left / 4):
                _set_timeout(conn, deadline)
                last_timeout = now
            rows = cursor.fetchmany(batch)
            if not rows:
                break
            # Preserve nullable 64-bit keys exactly. A float intermediary can merge distinct IDs.
            chunk = pd.DataFrame(rows, columns=cols, dtype=object)
            n_rows += len(chunk)
            n_bytes += int(chunk.memory_usage(deep=True).sum())
            if n_rows > _AUX_ROW_LIMIT or n_bytes > memory_limit:
                raise ValueError("превышен бюджет памяти справочника")
            batch = min(_fetch_size(n_rows, n_bytes), _AUX_ROW_LIMIT + 1 - n_rows)
            chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=cols)


def enrich_postgres(conn, tables, train, test, id_column, target_column, task_text,
                    output_dir, deadline=None, log=print, on_audit=None, replay=None, event_counts=None):
    """Add checked dimensions to the one-time CSV snapshot shared by all experiments."""
    import os
    import shutil
    import tempfile
    import time
    from leak_audit import sample_positions

    output_dir = Path(output_dir)
    train_only = (set(train.columns) - set(test.columns)) | {target_column}
    clock_descriptions = _dictionary_clocks(conn, tables, train, test, deadline, log)
    passes = [(table, False) for table in tables.values()]
    parents = []
    earliest = {}
    for table, indirect in passes:
        if deadline is not None and time.monotonic() >= deadline:
            if table not in (train, test):
                log(f"JOIN {table.schema}.{table.name} пропущен: истёк бюджет JOIN")
            continue
        detail = {"table": f"{table.schema}.{table.name}", "status": "skipped", "output_columns": [],
                  "dependencies": [], "source_columns": {col: table.column_types.get(col) for col in table.columns},
                  "primary_key": list(table.primary_key), "foreign_keys": [
                      {"columns": list(f.columns), "target": ".".join(f.target),
                       "target_columns": list(f.target_columns)} for f in table.foreign_keys]}
        try:
            routes = _bridge_routes(table, parents) if indirect else {}
            descriptions = clock_descriptions.get((table.schema, table.name), {})
            clock_fields = [c for c in table.columns if c.lower() in _START_NAMES | _END_NAMES | _GENERIC_HOUR_COLUMNS]
            if clock_fields:
                detail["clock_candidates"] = {c: {"catalog_comment_present": bool(table.comments.get(c)),
                                                  "dictionary_description_present": bool(descriptions.get(c))}
                                              for c in clock_fields}
            plan = _plan_relation(table, train, test, id_column, train_only, task_text, descriptions, routes)
            if plan is None:
                continue
            left, right, event_col, hour_col, mode, values = plan
            prefix = table.name + "__"
            detail["candidate_columns"] = values.copy()
            detail.update(left_key=left, right_key=right, source_clock=hour_col, event_clock=event_col,
                          availability=f"{mode}, available < event" if hour_col else "static_dimension")
            interval = mode in ("start", "end", "inferred")
            publication = _publication_clock(table, task_text) if interval else None
            if publication:
                detail.update(publication_clock=publication, publication_evidence="documented_first_immutable_publication",
                              availability=f"hour_{mode}, max(interval_end, publication) < event")
            if hour_col:
                detail["clock_evidence"] = ("data_dictionary" if _hour_clock(table, task_text)[0] is None
                                            else "column_name_catalog_or_task")
            if indirect:
                parent, via = routes[(left, right)]
                detail["dependencies"] = [f"{parent.schema}.{parent.name}"]
                detail["bridge_column"] = via
            cols = [right] + ([hour_col] if hour_col else []) + ([publication] if publication else []) + values
            with conn.transaction():
                auxiliary = _read_lookup(conn, table, cols, deadline, hourly=interval)
            if mode == "inferred" and event_col not in earliest:
                first = [to_utc_naive(chunk[event_col]).min()
                         for name in ("train.csv", "test.csv")
                         for chunk in pd.read_csv(output_dir / name, usecols=[event_col], dtype=str,
                                                  encoding="utf-8-sig", chunksize=_EVENT_CHUNK_ROWS)]
                earliest[event_col] = pd.Series(first, dtype="datetime64[ns]").min()
            hour_col, mode, reason, values = _reference_mode(
                table, auxiliary, right, hour_col, mode, earliest.get(event_col), publication, values)
            if not hour_col:
                event_col = None
            interval = mode in ("start", "end", "inferred")
            duplicate_rows = int(auxiliary[right].duplicated().sum())
            if on_audit is not None:
                detail["lookup_rows"] = len(auxiliary)
                detail["distinct_lookup_keys"] = int(auxiliary[right].astype("string").nunique())
            lookup = _prepare_lookup(auxiliary, right, hour_col, mode, values, prefix, publication, coalesce=True,
                                     cadence_before=earliest.get(event_col), diagnostics=detail)
            if mode == "inferred":
                detail["interval_step"] = str(lookup.attrs["inferred_step"])
                detail["clock_evidence"] = "within_entity_median_before_first_event"
            payload = [c for c in lookup if not c.startswith("__")]
            del auxiliary
            splits = {}
            samples = []
            with tempfile.TemporaryDirectory(dir=output_dir) as staging:
                for name in ("train.csv", "test.csv"):
                    split_samples = []
                    stats = {"rows": 0, "matched_rows": 0, "missing_key_rows": 0}
                    if on_audit is not None:
                        stats["feature_nonmissing"] = dict.fromkeys(payload, 0)
                    splits[Path(name).stem] = stats
                    dest = Path(staging) / name
                    chosen = sample_positions(event_counts[Path(name).stem]) if replay and event_counts else None
                    with pd.read_csv(output_dir / name, dtype=str, keep_default_na=False,
                                     encoding="utf-8-sig", chunksize=_EVENT_CHUNK_ROWS) as chunks:
                        for i, events in enumerate(chunks):
                            if deadline is not None and time.monotonic() >= deadline:
                                raise TimeoutError("истёк бюджет JOIN")
                            joined = _join_chunk(events, lookup, left, event_col, stats)
                            if (not joined.index.equals(events.index) or
                                    (id_column in events and not joined[id_column].equals(events[id_column]))):
                                raise ValueError("изменились строки событий")
                            if chosen is not None and len(events):
                                ids = chosen[(chosen >= events.index[0]) & (chosen <= events.index[-1])]
                                if len(ids):
                                    split_samples.append((events.loc[ids], joined.loc[ids, joined.columns.difference(events.columns, sort=False)]))
                            joined.to_csv(dest, index=False, mode="a", header=(i == 0), encoding="utf-8")
                    if split_samples:
                        samples.append(tuple(pd.concat(parts) for parts in zip(*split_samples)))
                if replay and samples:
                    _audit_lookup(samples, lookup, left, event_col, replay, detail["table"])
                # Preserve the original inode until both compatible exports are installed.
                train_path = output_dir / "train.csv"
                original = Path(staging) / "original-train.csv"
                try:
                    os.link(train_path, original)
                except OSError:
                    shutil.copyfile(train_path, original)
                (Path(staging) / "train.csv").replace(train_path)
                try:
                    (Path(staging) / "test.csv").replace(output_dir / "test.csv")
                except BaseException:
                    original.replace(train_path)
                    raise
            detail.update(status="applied", output_columns=payload, **splits)
            if not interval and not indirect and not event_col:
                parents.append((table, values))
            rule = (f", {hour_col}: start+1h < {event_col}" if mode == "start" else
                    f", {mode} clock {hour_col} -> {event_col}" if hour_col else ", one profile per key")
            if mode == "inferred":
                rule = f", {hour_col} + {detail['interval_step']} < {event_col}"
            extra = f", additional profile rows={duplicate_rows}" if not interval else ""
            counts = ", ".join(f"{split}={stats['matched_rows']}/{stats['rows']}" for split, stats in splits.items())
            selection = ("latest version strictly before event" if mode in ("profile", "dated_profile", "dated_static")
                         else "deterministic key/payload order; conflicting values masked")
            profile = f", selection={selection}" if not interval else ""
            display_mode = {"dated_static": "static", "dated_profile": "profile"}.get(mode, mode or "static")
            diagnostics = (f" step={detail['interval_step']} duplicate_timestamps={detail['duplicate_timestamps']}"
                           if mode == "inferred" else "")
            log(f"JOIN {table.schema}.{table.name}: {left} -> {right}{rule}{extra}{profile}, {counts}, "
                f"rows_and_ids_preserved=True, source_features={values!r}, mode={display_mode} reason={reason}{diagnostics}")
        except _NoEventRelation as exc:
            if not indirect:
                passes.append((table, True))
                continue
            detail["reason"] = str(exc)
            log(f"JOIN {table.schema}.{table.name} пропущен: {exc}")
        except (ValueError, TimeoutError) as exc:
            detail["reason"] = str(exc)
            duplicates = (f" duplicate_timestamps={detail['duplicate_timestamps']}"
                          if "duplicate_timestamps" in detail else "")
            log(f"JOIN {table.schema}.{table.name} skipped reason={exc}{duplicates}")
        except Exception as exc:
            # SQL errors may contain row contents or connection parameters.
            detail["reason"] = type(exc).__name__
            log(f"JOIN {table.schema}.{table.name} пропущен ({type(exc).__name__})")
        if on_audit is not None:
            on_audit(detail)
