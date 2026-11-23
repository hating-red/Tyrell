import csv
import math
import re
from pathlib import Path

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
    excluded = {target, identifier, *targets, *declared_ids}
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
        past = pd.to_datetime(left, format="mixed", errors="coerce", utc=True)
        future = pd.to_datetime(right, format="mixed", errors="coerce", utc=True)
        if past.notna().all() and future.notna().all() and past.nunique() > 1 and past.max() < future.min():
            candidates.append(col)
    return candidates[0] if len(candidates) == 1 else None


def detect_split(X_train, X_test, max_groups=50, min_unseen=0.9, time_column=None):
    """Определяет временную ось событий или колонку независимых групп.

    Приоритет у времени события и числовых периодов, когда test следует за train.
    Для остальных колонок проверяем долю новых групп в test. Возвращает словарь
    с типом time/group и именем колонки или None для обычной валидации.
    """
    clock = detect_time_column(X_train, X_test, time_column)
    if clock is not None:
        return {"type": "time", "column": clock}
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
    if task_type == "regression" and (target.lower() == "response_time_minutes" or
            re.search(r"non[- ]?negative|неотрицательн", task_text, re.I)):
        spec["nonnegative"] = True
    if task_type == "regression":
        declared_units = re.findall(r"^\s*label delay unit\s*:\s*[`]?([\w-]+)[`]?\.?\s*$",
                                    task_text, re.I | re.M)
        response_duration = (target.lower() == "response_time_minutes" and
                             re.search(r"(?:число|количество)\s+минут\s+от\s+регистрации\s+вызова\s+"
                                       r"до\s+прибытия\s+(?:экстренной\s+)?службы\b", task_text, re.I))
        if ({unit.lower() for unit in declared_units} == {"minutes"} or
                (not declared_units and response_duration)):
            spec["label_delay_unit"] = "minutes"
    return spec, confident


# These tables are documented static dimensions. Other schemas can declare the
# relationship through foreign keys. Hourly sources additionally need an interval clock.
_DIMENSION_KEYS = {"ads": "ad_id", "campaigns": "campaign_id",
                   "users": "user_id", "districts": "district_id"}
_HOURLY_TABLES = {"weather", "traffic", "air_quality", "power_grid", "public_transport"}
_START_NAMES = {"hour_start", "hour_start_time", "hour_start_ts", "hour_start_at", "interval_start",
                "interval_start_time", "period_start", "bucket_start"}
_END_NAMES = {"hour_end", "hour_end_time", "hour_end_ts", "hour_end_at", "interval_end",
              "interval_end_time", "period_end", "bucket_end"}
_EVENT_CHUNK_ROWS = 20_000
_AUX_ROW_LIMIT = 1_000_000
_AUX_MEMORY_LIMIT = 192 * 1024**2


_GENERIC_HOUR_COLUMNS = {"timestamp", "datetime", "hour", "hour_ts", "hour_timestamp", "ts", "hourly_timestamp"}


def _dictionary_clocks(conn, tables, train, test, deadline, log):
    """Read bounded, explicitly attributed clock descriptions in the current snapshot."""
    import time
    from postgres_loader import _identifier, _relation, _set_timeout

    candidates = [t for t in tables.values() if t not in (train, test) and t.name.lower() != "data_dictionary"
                  and any(c.lower() in _GENERIC_HOUR_COLUMNS for c in t.columns)]
    if not candidates:
        return {}
    names = sorted({f"{t.schema}.{t.name}" for t in candidates} | {t.name for t in candidates})
    clock_names = sorted({c for t in candidates for c in t.columns if c.lower() in _GENERIC_HOUR_COLUMNS})
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
    for column in table.columns:
        name = column.lower()
        if name not in _START_NAMES | _END_NAMES | _GENERIC_HOUR_COLUMNS:
            continue
        if name in _START_NAMES:
            # An arbitrary interval can be longer than an hour on a private schema.
            if ("hour" in name or table.name.lower() in _HOURLY_TABLES or
                    "hourly" in table.name.lower() or
                    re.search(r"\bhour(?:ly)?\b|часов", table.comments.get(column) or "", re.I)):
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
    # Prefer the published interval end when both bounds are present.
    for mode in ("end", "start"):
        matches = [col for col, value in clocks.items() if value == mode]
        if len(matches) == 1:
            return matches[0], mode
        if matches:
            break
    return None, None


def _publication_clock(table, task_text):
    """A plain timestamp or last-update clock cannot reconstruct first-published payloads."""
    names = {c.lower(): c for c in table.columns}
    if names.keys() & {"version", "revision", "version_id", "revision_id", "revised_at", "updated_at",
                       "modified_at", "valid_from", "valid_to", "is_current", "snapshot_date", "snapshot_at"}:
        raise ValueError("не подтверждена история публикации и исправлений измерений")
    publication = [names[k] for k in ("available_at", "published_at", "released_at", "publication_time",
                                      "publish_time", "ingested_at") if k in names]
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
        if not declared and table.schema == parent.schema:
            key = (_DIMENSION_KEYS.get(table.name.lower()) or
                   (table.primary_key[0] if len(table.primary_key) == 1 else None))
            if key and key.lower().endswith("_id") and key in table.columns:
                links = {(key, key)}
        for via, right in links:
            if via in fields and right in table.columns:
                routes[(parent.name + "__" + via, right)] = parent, via
    return routes


def _profile_bridge_key(conn, table, key, deadline):
    """Measure only cardinality of an unusable bridge; never read its payload."""
    import time
    from postgres_loader import _identifier, _relation, _set_timeout

    stop = time.monotonic() + 3
    sql = (f"SELECT count(*), count(k), count(DISTINCT k) FROM "
           f"(SELECT {_identifier(key)} AS k FROM {_relation(table)} LIMIT {_AUX_ROW_LIMIT + 1}) AS keys")
    try:
        with conn.transaction():
            _set_timeout(conn, min(deadline, stop) if deadline is not None else stop)
            total, nonnull, distinct = conn.execute(sql).fetchone()
    except Exception as exc:
        return {"status": "unavailable", "reason": type(exc).__name__}
    if total > _AUX_ROW_LIMIT:
        return {"status": "row_limit"}
    return {"status": "complete", "rows": total, "null_key_rows": total - nonnull, "distinct_keys": distinct,
            "unique_and_nonnull": total == nonnull == distinct}


def _pre_event_fields(table, values, task_text):
    """Additional paths and repaired dimensions require explicit feature-level evidence."""
    accepted = {}
    for col in values:
        full = f"{table.schema}.{table.name}.{col}"
        comment = str(table.comments.get(col) or "").strip()
        if re.search(r"(?:^|_)(?:future|next|last|latest|total|cumulative|cum|outcomes?|clicks?|ctr|"
                     r"responses?|conversions?|purchases?|engagement|impressions?|realized)(?:_|$)", col, re.I):
            continue
        # A declaration can document provenance on private schemas without code changes.
        documented = re.search(rf"^\s*Pre-event static feature\s*:\s*{re.escape(full)}\s*$",
                               task_text.replace(chr(96), ""), re.M | re.I)
        uncertain = re.search(r"\b(?:not|after|unknown|may|future|until|updated|нет|не|после)\b|\?", comment, re.I)
        if uncertain:
            continue
        if documented:
            accepted[col] = "task_pre_event_static_feature"
        elif (re.search(r"\b(?:static|immutable|fixed)\b|статическ|неизмен", comment, re.I)
              and re.search(r"(?:known|available|recorded|fixed) (?:at creation|before (?:the |each )?event)"
                            r"|(?:извест|доступ|задан|фиксир)\w* (?:при создании|до события)", comment, re.I)):
            accepted[col] = "catalog_pre_event_static_feature"
    return accepted


def _collapse_static_rows(frame, key, values):
    columns = [key, *values]
    keys = frame[key].astype("string")
    present = keys.notna() & keys.str.strip().ne("")
    records = frame.loc[present, columns].copy()
    records[key] = keys.loc[present]
    duplicates = records[key].duplicated(keep=False)
    distinct = records.drop_duplicates()
    # Null vs known is a conflict too. Keep an attribute only if every copy agrees.
    conflicts = distinct.groupby(key, sort=False)[values].nunique(dropna=False).gt(1)
    unique = distinct.drop_duplicates(subset=key).set_index(key)
    safe = unique.mask(conflicts).loc[~conflicts.all(axis=1)].reset_index()
    return safe, {"missing_lookup_key_rows": int((~present).sum()),
                  "duplicate_keys": int(records.loc[duplicates, key].nunique()),
                  "identical_extra_rows": len(records) - len(distinct),
                  "conflicts_per_column": {c: int(conflicts[c].sum()) for c in values},
                  "omitted_conflicting_keys": int(conflicts.all(axis=1).sum()),
                  "retained_keys": len(safe)}


def _plan_relation(table, train, test, identifier, train_only, task_text, descriptions=None, bridges=None):
    common = set(train.columns).intersection(test.columns)
    name = table.name.lower()
    if (table in (train, test) or "submission" in name or
            re.search(r"(?:^|_)(?:labels?|targets?|outcomes?|train|test)(?:_|$)", name)):
        return None
    if name == "data_dictionary":
        raise ValueError("словарь метаданных")
    hour_col, mode = _hour_clock(table, task_text, descriptions)
    names = {c.lower() for c in table.columns}
    hourly = bool(hour_col or name in _HOURLY_TABLES or "hourly" in name or
                  names & (_START_NAMES | _END_NAMES | _GENERIC_HOUR_COLUMNS))
    if hourly and not hour_col:
        raise ValueError("не подтверждены начало/конец часа")
    publication = _publication_clock(table, task_text) if hourly else None
    event_col = (_task_clock(task_text) or _named_event_clock(common)) if hourly else None
    if hourly and event_col not in common:
        raise ValueError("не подтверждено время события")

    relations = {(fk.columns[0], fk.target_columns[0])
                 for main in (train, test) for fk in main.foreign_keys
                 if tuple(fk.target) == (table.schema, table.name) and len(fk.columns) == 1
                 and len(fk.target_columns) == 1 and fk.columns[0] in common}
    if not relations and train.schema == test.schema == table.schema:
        key = _DIMENSION_KEYS.get(name)
        if not key and len(table.primary_key) == 1:
            primary = table.primary_key[0]
            if primary.lower().endswith("_id") and primary in common:
                key = primary
        if hourly:
            # Only an unambiguous common entity key may address an hourly history.
            keys = [col for col in table.columns if col in common and col != identifier
                    and col.lower().endswith("_id")]
            key = keys[0] if len(keys) == 1 else None
        if key and key in common and key in table.columns:
            relations.add((key, key))
    if not relations and bridges and not hourly:
        relations = set(bridges)
    if not relations:
        raise _NoEventRelation("нет однозначной связи с событиями")
    if len(relations) != 1:
        raise ValueError("неоднозначный путь к справочнику")
    left, right = relations.pop()
    if left == identifier or left in train_only:
        raise ValueError("ключ совпадает с идентификатором события или результатом")
    if hourly:
        event_type = train.column_types.get(event_col)
        test_type = test.column_types.get(event_col)
        source_type = table.column_types.get(hour_col)
        if event_type and test_type and source_type:
            clocks = {t == "timestamp with time zone" for t in (event_type, test_type, source_type)}
            if len(clocks) > 1:
                raise ValueError("не согласованы часовые пояса источника и событий")
    if not hourly and any(("timestamp" in table.column_types.get(c, "").lower()
                           or c.lower() in {"valid_from", "valid_to", "as_of", "version", "snapshot_date"}
                           or c.lower().endswith(("_time", "_at", "_timestamp")))
                          for c in table.columns):
        raise ValueError("у справочника есть временные поля без правила доступности")

    blocked = {x.lower() for x in train_only} | {identifier.lower(), "label", "target", "outcome",
                                                 "prediction", "predicted", "submission"}
    unsupported_types = {"bytea", "json", "jsonb", "array", "user-defined"}
    values = []
    for c in table.columns:
        low = c.lower()
        if (c == right or c == hour_col or c == publication or low in _START_NAMES | _END_NAMES or
                any(re.search(rf"(?:^|_){re.escape(term)}(?:_|$)", low) for term in blocked)
                or table.column_types.get(c, "").lower() in unsupported_types):
            continue
        values.append(c)
    if not values:
        raise ValueError("нет доступных признаков")
    return left, right, event_col, hour_col, mode, values


def label_available_at(times, y_fit, unit):
    import numpy as np

    if unit != "minutes" or not pd.api.types.is_datetime64_any_dtype(times):
        raise ValueError("A duration label needs a datetime event clock and unit 'minutes'")
    duration = np.asarray(y_fit, dtype=float)
    if np.any(duration < 0) or np.isinf(duration).any():
        raise ValueError("Label delays must be nonnegative and finite")
    return times + pd.to_timedelta(duration, unit="m")


def parse_event_datetime(series):
    """Reject clock-only strings: a calendar date filled in at run time is not event evidence."""
    if not pd.api.types.is_datetime64_any_dtype(series):
        text = series.astype("string").str.strip()
        clock_only = text.str.fullmatch(r"\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?\s*(?:AM|PM)?\s*(?:Z|[+-]\d{2}:?\d{2})?",
                                       flags=re.I, na=False)
        series = series.mask(clock_only)
    return pd.to_datetime(series, errors="coerce", utc=True, format="mixed")


def _asof_clock(series):
    if pd.api.types.is_numeric_dtype(series):
        raise ValueError("не подтверждена единица времени числовой метки")
    times = parse_event_datetime(series)
    if not times.notna().any():
        return times, None
    if pd.api.types.is_datetime64_any_dtype(series):
        return times, isinstance(series.dtype, pd.DatetimeTZDtype)
    lexemes = series[times.notna()].astype(str)
    zoned = lexemes.str.contains(r"(?:[zZ]|[+-]\d{2}:?\d{2})\s*$")
    if zoned.any() and not zoned.all():
        raise ValueError("смешаны метки времени с часовым поясом и без него")
    return times, bool(zoned.all())


def _prepare_lookup(auxiliary, right, hour_col, mode, values, prefix, publication=None):
    """Validate keys after normalisation, before any train/test rows are touched."""
    keys = ["__entity", "__available"] if hour_col else ["__entity"]
    lookup = auxiliary[values].add_prefix(prefix).copy()
    lookup["__entity"] = auxiliary[right].astype("string")
    if lookup["__entity"].isna().any() or lookup["__entity"].str.strip().eq("").any():
        raise ValueError("пропущен ключ справочника")
    if hour_col:
        if mode not in {"start", "end"}:
            raise ValueError("не подтверждены начало/конец часа")
        stamps, zoned = _asof_clock(auxiliary[hour_col])
        if stamps.isna().any() or (stamps != stamps.dt.floor("h")).any():
            raise ValueError("не определена часовая граница интервала")
        lookup["__available"] = stamps + (pd.Timedelta(hours=1) if mode == "start" else pd.Timedelta(0))
    if lookup.duplicated(keys).any():
        raise ValueError("неуникальный ключ справочника")
    if hour_col:
        if publication:
            released, release_zone = _asof_clock(auxiliary[publication])
            if released.isna().any() or release_zone != zoned:
                raise ValueError("не определено время или часовой пояс публикации")
            lookup["__interval_end"] = lookup["__available"]
            lookup["__available"] = lookup["__interval_end"].where(lookup["__interval_end"] >= released, released)
            lookup = lookup.sort_values(["__available", "__interval_end"], kind="stable")
            lookup = lookup.drop_duplicates(["__entity", "__available"], keep="last")
            # A late-arriving older hour can never replace a more recent known interval.
            latest = lookup.groupby("__entity", sort=False)["__interval_end"].cummax()
            lookup = lookup.loc[lookup["__interval_end"].eq(latest)].reset_index(drop=True)
        else:
            lookup = lookup.sort_values("__available", kind="stable").reset_index(drop=True)
        lookup.attrs["zoned"] = zoned
        return lookup
    return lookup.set_index("__entity")


def _join_chunk(events, lookup, left, event_col, audit=None):
    payload = [c for c in lookup if c not in ("__entity", "__available", "__interval_end")]
    if set(events).intersection(payload):
        raise ValueError("совпадающие имена признаков")
    keys = events[left].astype("string")
    if event_col:
        times, zoned = _asof_clock(events[event_col])
        if zoned is not None and lookup.attrs["zoned"] is not None and zoned != lookup.attrs["zoned"]:
            raise ValueError("не согласованы часовые пояса источника и событий")
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
    else:
        aligned = lookup.reindex(keys)[payload]
        if audit is not None:
            matched = keys.isin(lookup.index)
    if audit is not None:
        audit["rows"] += len(events)
        audit["matched_rows"] += int(matched.sum())
        audit["missing_key_rows"] += int((keys.isna() | keys.str.strip().eq("")).sum())
        for col in payload:
            audit["feature_nonmissing"][col] += int((aligned[col].notna() & aligned[col].ne("")).sum())
    aligned.index = events.index
    return pd.concat([events, aligned], axis=1)


def _read_lookup(conn, table, cols, deadline):
    import time
    from postgres_loader import _fetch_size, _identifier, _relation, _remaining, _set_timeout

    # A server-side cursor prevents a large dimension from being buffered by libpq.
    fields = ", ".join(map(_identifier, cols))
    query = f"SELECT {fields} FROM {_relation(table)} LIMIT {_AUX_ROW_LIMIT + 1}"
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
            if n_rows > _AUX_ROW_LIMIT or n_bytes > _AUX_MEMORY_LIMIT:
                raise ValueError("превышен бюджет памяти справочника")
            batch = min(_fetch_size(n_rows, n_bytes), _AUX_ROW_LIMIT + 1 - n_rows)
            chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=cols)


def enrich_postgres(conn, tables, train, test, id_column, target_column, task_text,
                    output_dir, deadline=None, log=print, on_audit=None):
    """Add checked dimensions to the one-time CSV snapshot shared by all experiments."""
    import os
    import shutil
    import tempfile
    import time

    output_dir = Path(output_dir)
    train_only = (set(train.columns) - set(test.columns)) | {target_column}
    clock_descriptions = _dictionary_clocks(conn, tables, train, test, deadline, log)
    passes = [(table, False) for table in tables.values()]
    parents = []
    for table, indirect in passes:
        if deadline is not None and time.monotonic() >= deadline:
            log("Достигнут лимит времени JOIN")
            break
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
                          availability=f"hour_{mode}, interval_end < event" if hour_col else "static_dimension")
            publication = _publication_clock(table, task_text) if hour_col else None
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
                detail["bridge_availability"] = _pre_event_fields(parent, [via], task_text).get(via)
                evidence = _pre_event_fields(table, values, task_text)
                detail["static_evidence"] = evidence
                values = list(evidence)
                if not detail["bridge_availability"] or not values:
                    if on_audit is not None:
                        detail["key_profile"] = _profile_bridge_key(conn, table, right, deadline)
                    if not detail["bridge_availability"]:
                        raise ValueError("не подтверждена доступность промежуточного ключа до события")
                    raise ValueError("не подтверждена доступность статических признаков до события")
            cols = [right] + ([hour_col] if hour_col else []) + ([publication] if publication else []) + values
            with conn.transaction():
                auxiliary = _read_lookup(conn, table, cols, deadline)
            if on_audit is not None:
                detail["lookup_rows"] = len(auxiliary)
                detail["distinct_lookup_keys"] = int(auxiliary[right].astype("string").nunique())
            if not hour_col and auxiliary[right].astype("string").duplicated().any():
                evidence = _pre_event_fields(table, values, task_text)
                detail["static_evidence"] = evidence
                # Audit conflicts without using unaudited payloads as features.
                eligible = list(evidence)
                auxiliary, detail["duplicate_resolution"] = _collapse_static_rows(auxiliary, right, eligible or values)
                if not eligible or auxiliary.empty:
                    raise ValueError("неуникальный ключ справочника без безопасных статических значений")
                values = eligible
            payload = [prefix + c for c in values]
            lookup = _prepare_lookup(auxiliary, right, hour_col, mode, values, prefix, publication)
            del auxiliary
            splits = {}
            with tempfile.TemporaryDirectory(dir=output_dir) as staging:
                for name in ("train.csv", "test.csv"):
                    stats = None
                    if on_audit is not None:
                        stats = {"rows": 0, "matched_rows": 0, "missing_key_rows": 0,
                                 "feature_nonmissing": dict.fromkeys(payload, 0)}
                        splits[Path(name).stem] = stats
                    dest = Path(staging) / name
                    with pd.read_csv(output_dir / name, dtype=str, keep_default_na=False,
                                     encoding="utf-8-sig", chunksize=_EVENT_CHUNK_ROWS) as chunks:
                        for i, events in enumerate(chunks):
                            if deadline is not None and time.monotonic() >= deadline:
                                raise TimeoutError("истёк бюджет JOIN")
                            joined = _join_chunk(events, lookup, left, event_col, stats)
                            if (not joined.index.equals(events.index) or
                                    (id_column in events and not joined[id_column].equals(events[id_column]))):
                                raise ValueError("изменились строки событий")
                            joined.to_csv(dest, index=False, mode="a", header=(i == 0), encoding="utf-8")
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
            if not hour_col and not indirect:
                parents.append((table, values))
            rule = f", {hour_col} ({mode}), interval_end < {event_col}" if hour_col else ""
            log(f"JOIN {table.schema}.{table.name}: {len(values)} признаков, {left} -> {right}{rule}")
        except _NoEventRelation as exc:
            if not indirect:
                passes.append((table, True))
                continue
            detail["reason"] = str(exc)
            log(f"JOIN {table.schema}.{table.name} пропущен: {exc}")
        except (ValueError, TimeoutError) as exc:
            detail["reason"] = str(exc)
            log(f"JOIN {table.schema}.{table.name} пропущен: {exc}")
        except Exception as exc:
            # SQL errors may contain row contents or connection parameters.
            detail["reason"] = type(exc).__name__
            log(f"JOIN {table.schema}.{table.name} пропущен ({type(exc).__name__})")
        if on_audit is not None:
            on_audit(detail)
