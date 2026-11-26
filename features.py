"""Автоматические признаки без LLM.

Простые преобразования, полезные почти на любой таблице: части дат, частоты категорий,
число пропусков в строке, средние и отклонения важных числовых признаков по категориям.
При временной валидации используются только сведения из текущей строки.
Полезность набора проверяется снаружи на тех же фолдах, что и остальные эксперименты.
"""

import numpy as np
import pandas as pd

MAX_GROUP_COLUMNS = 3             # сколько категорий берем для групповых признаков
MAX_GROUP_NUMERIC = 5             # сколько числовых признаков агрегируем по группам
MAX_GROUP_CARDINALITY = 1000      # категории с большим числом значений для групп не годятся
ENTITY_MAX_COLUMNS = 6            # из скольких колонок максимум собираем ключ сущности
ENTITY_CANDIDATES = 40            # сколько колонок перебираем при поиске ключа
ENTITY_KEEP_REPEAT = 0.95         # колонка в ключе почти не должна снижать повторяемость сущности
ENTITY_KEEP_COVERAGE = 0.9        # и покрытие test
TE_PREFIX = "__te_"
DATE_PATTERN = r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}"


def _is_categorical(s):
    return (isinstance(s.dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(s)
            or pd.api.types.is_string_dtype(s))


def _is_numeric(s):
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def _parse_dates(s):
    """Строковая колонка с датами в datetime, иначе None."""
    from data_tools import to_utc_naive
    if pd.api.types.is_datetime64_any_dtype(s):
        return to_utc_naive(s)
    if not _is_categorical(s):
        return None
    sample = s.dropna().astype(str).head(500)
    if sample.empty or sample.str.contains(DATE_PATTERN).mean() < 0.9:
        return None
    dates = to_utc_naive(s)
    return dates if dates.notna().sum() >= 0.9 * s.notna().sum() else None


def important_numeric(X, y, task_type, limit=MAX_GROUP_NUMERIC, seed=42):
    """Самые важные числовые признаки по быстрой модели LightGBM."""
    import lightgbm as lgb

    numeric = [c for c in X if _is_numeric(X[c])]
    if len(numeric) <= limit:
        return numeric
    frame = X.copy()
    for column in frame:
        if _is_categorical(frame[column]):
            frame[column] = frame[column].astype("category")
    labeled = pd.Series(np.asarray(y)).notna().to_numpy()
    target = pd.Series(np.asarray(y))[labeled]
    params = {"n_estimators": 200, "learning_rate": 0.1, "num_leaves": 31, "verbose": -1,
              "importance_type": "gain", "random_state": seed, "n_jobs": 8}
    if task_type == "regression":
        model = lgb.LGBMRegressor(**params)
        target = pd.to_numeric(target, errors="coerce")
    else:
        model = lgb.LGBMClassifier(**params)
        target = target.astype(str)
    model.fit(frame[labeled], target)
    gain = pd.Series(model.feature_importances_, index=frame.columns)
    return gain[numeric].sort_values(ascending=False).index[:limit].tolist()


def add_auto_features(X_train, X_test, numeric_priority=None, *, row_local=False):
    """Возвращает X_train и X_test с новыми колонками; исходные колонки не меняются."""
    full = pd.concat([X_train, X_test], ignore_index=True)
    new = {}

    categorical = [c for c in full if _is_categorical(full[c])]
    if row_local:
        from data_tools import _EVENT_CLOCKS, parse_event_datetime
    for column in full.columns:
        if row_local:
            series = full[column]
            if pd.api.types.is_datetime64_any_dtype(series):
                dates = series
            elif column.lower() in _EVENT_CLOCKS and not pd.api.types.is_numeric_dtype(series):
                dates = parse_event_datetime(series)
            else:
                continue
        else:
            dates = _parse_dates(full[column])
        if dates is None:
            continue
        if column in categorical:
            categorical.remove(column)
        new[f"{column}__year"] = dates.dt.year
        new[f"{column}__month"] = dates.dt.month
        new[f"{column}__dayofweek"] = dates.dt.dayofweek
        new[f"{column}__weekend"] = dates.dt.dayofweek.ge(5).astype("float32").where(dates.notna())
        new[f"{column}__dayofyear"] = dates.dt.dayofyear
        if row_local or (dates.dt.hour.fillna(0) != 0).any():
            new[f"{column}__hour"] = dates.dt.hour
        if row_local:
            new[f"{column}__hour_of_week"] = dates.dt.dayofweek * 24 + dates.dt.hour

    for column in ([] if row_local else categorical):
        counts = full[column].value_counts()
        new[f"{column}__freq"] = full[column].map(counts).astype(float)

    missing = full.isna().sum(axis=1)
    if row_local or missing.any():
        new["__missing_count"] = missing.astype(float)

    groups = [] if row_local else [c for c in categorical if 2 <= X_train[c].nunique() <= MAX_GROUP_CARDINALITY]
    groups = sorted(groups, key=lambda c: -X_train[c].nunique())[:MAX_GROUP_COLUMNS]
    numeric = [c for c in (numeric_priority or []) if c in full and _is_numeric(full[c])]
    if not numeric:
        numeric = [c for c in full if _is_numeric(full[c])][:MAX_GROUP_NUMERIC]
    for group in groups:
        for column in numeric:
            mean = full.groupby(group, observed=True)[column].transform("mean")
            new[f"{column}__mean_by_{group}"] = mean
            new[f"{column}__diff_by_{group}"] = full[column] - mean

    if not new:
        return X_train, X_test
    added = pd.DataFrame(new).reset_index(drop=True)
    added = added.loc[:, [c for c in added if c not in full.columns]]
    result = pd.concat([full.reset_index(drop=True), added], axis=1)
    n = len(X_train)
    return result.iloc[:n].reset_index(drop=True), result.iloc[n:].reset_index(drop=True)


def temporal_entity_candidates(frame, clock):
    """Select repeated identifiers in the initial model-fit prefix only."""
    history, numeric_ids = [], []
    for name, values in frame.items():
        if name == clock or not str(name).lower().endswith(("_id", "_code", "_key")):
            continue
        observed = values.dropna()
        distinct = observed.nunique()
        if not 1 <= distinct < len(observed):
            continue
        if (pd.api.types.is_datetime64_any_dtype(frame[clock])
                and not any(name + suffix in frame for suffix in ("__past_events", "__since_previous_s"))):
            history.append(name)
        if _is_numeric(values) and len(numeric_ids) < 6:
            if pd.api.types.is_integer_dtype(values) or (np.isfinite(observed) & (observed % 1 == 0)).all():
                numeric_ids.append(name)
    return {"entity_history": history, "categorical_ids": numeric_ids}


def add_fold_entity_history(fit, queries, columns, clock):
    """Count strictly earlier fit events and measure elapsed seconds since the last one.

    The batch of queries may contain a monitor, outer validation and test. Its rows
    never enter the history. All events with the same timestamp see the same past.
    """
    fit, queries = fit.copy(), queries.copy()
    all_times = pd.concat([fit[clock], queries[clock]], ignore_index=True)
    for col in columns:
        fit_codes, vocabulary = pd.factorize(fit[col], sort=False)
        fit_codes = fit_codes.astype("int64", copy=False)
        query_codes = pd.Index(vocabulary).get_indexer(queries[col]).astype("int64", copy=False)
        past = pd.DataFrame({"key": fit_codes, "when": all_times.iloc[:len(fit)].array})
        past = past.loc[past.key.ge(0) & past.when.notna()]
        history = (past.groupby(["key", "when"], sort=False).size().rename("events")
                   .reset_index().sort_values("when").rename(columns={"when": "previous"}))
        history["events"] = history.groupby("key", sort=False)["events"].cumsum()

        # Keep positions separate from DataFrame indices, which can be unordered or repeated.
        keys = np.concatenate([fit_codes, query_codes])
        query = pd.DataFrame({"key": keys, "when": all_times, "position": np.arange(len(all_times))})
        query = query.loc[query.when.notna()].sort_values("when")
        hits = pd.merge_asof(query, history, left_on="when", right_on="previous", by="key",
                             direction="backward", allow_exact_matches=False)
        counts = np.full(len(all_times), np.nan)
        gaps = np.full(len(all_times), np.nan)
        positions = hits["position"].to_numpy()
        counts[positions] = hits.events.fillna(0).to_numpy()
        gaps[positions] = (hits.when - hits.previous).dt.total_seconds().to_numpy()
        available = np.searchsorted(np.sort(fit[clock].dropna().to_numpy()), all_times.to_numpy(), side="left")
        frequency = counts / np.maximum(1, available)
        for name, values in ((col + "__past_events", counts), (col + "__past_frequency", frequency),
                             (col + "__since_previous_s", gaps)):
            fit[name] = values[:len(fit)]
            queries[name] = values[len(fit):]
    return fit, queries


def ad_history_keys(columns):
    """Find advertising entities in the event inputs and their joined attributes."""
    def find(*names):
        for name in names:
            if name in columns:
                return name
            matches = [column for column in columns if column.endswith("__" + name)]
            if len(matches) == 1:
                return matches[0]
        return None

    user, ad, campaign = find("user_id"), find("ad_id"), find("campaign_id")
    category = find("ad_type", "ad_category", "category")
    if user is None or ad is None:
        return {}
    groups = {"user": (user,), "ad": (ad,), "user_ad": (user, ad)}
    if campaign is not None:
        groups.update(campaign=(campaign,), user_campaign=(user, campaign))
    if category is not None:
        groups["user_ad_type"] = (user, category)
    return groups


def add_ad_history(fit, queries, labels, clock, groups, *, window, smoothing):
    """Compute engagement history using only strictly earlier rows of this fit prefix."""
    n_fit = len(fit)
    rows = pd.concat([fit, queries], ignore_index=True)
    labels = np.asarray(labels, dtype=float)
    timestamps = rows[clock].to_numpy()
    order = np.argsort(timestamps[:n_fit], kind="stable")
    prefix = np.r_[0., np.cumsum(labels[order])]
    counts = np.searchsorted(timestamps[:n_fit][order], timestamps, side="left")
    prior = np.divide(prefix[counts], counts, out=np.full(len(rows), np.nan), where=counts > 0)
    prior[pd.isna(timestamps)] = np.nan
    ordered_query = pd.DataFrame({"event": timestamps, "row": np.arange(len(rows))})
    ordered_query = ordered_query.dropna(subset=["event"]).sort_values("event", kind="stable")
    features = {}

    for name, cols in groups.items():
        keys = rows[list(cols)]
        codes, _ = pd.factorize(pd.MultiIndex.from_frame(keys), sort=False)
        codes = np.asarray(codes, dtype=np.int64)
        codes[~keys.notna().all(axis=1).to_numpy()] = -1
        history = pd.DataFrame({"entity": codes[:n_fit], "observed": timestamps[:n_fit], "label": labels})
        history = history.loc[history.entity.ge(0) & history.observed.notna()].sort_values("observed", kind="stable")
        grouped = history.groupby("entity", sort=False)
        history["events"] = grouped.cumcount() + 1
        history["positive"] = grouped["label"].cumsum()
        if name == "user":
            history["recent_positive"] = history["positive"] - grouped["positive"].shift(window, fill_value=0.)
            quoted = history["observed"].where(history["label"].eq(1))
            history["last_positive"] = quoted.groupby(history["entity"], sort=False).ffill()
        query = ordered_query.assign(entity=codes[ordered_query["row"].to_numpy()])
        matches = pd.merge_asof(query, history, by="entity", left_on="event", right_on="observed",
                                direction="backward", allow_exact_matches=False)
        aligned = matches.set_index("row").reindex(range(len(rows)))
        n = aligned["events"].fillna(0).to_numpy()
        positive = aligned["positive"].fillna(0).to_numpy()
        if name == "user_ad":
            features["__adh_user_ad_past_impressions"] = n - positive
            continue
        features[f"__adh_{name}_events"] = n
        features[f"__adh_{name}_rate"] = (positive + smoothing * prior) / (n + smoothing)
        if name == "user":
            recent_n = np.minimum(n, window)
            features[f"__adh_user_last_{window}_rate"] = np.divide(
                aligned["recent_positive"].to_numpy(), recent_n, out=np.full(len(rows), np.nan), where=recent_n > 0)
            features["__adh_user_seconds_since_engaged"] = (
                aligned["event"] - aligned["last_positive"]).dt.total_seconds().to_numpy()

    history = pd.DataFrame(features, dtype="float32")
    return (
        pd.concat([fit, history.iloc[:n_fit].set_axis(fit.index)], axis=1),
        pd.concat([queries, history.iloc[n_fit:].set_axis(queries.index)], axis=1),
    )


def _hash_rows(frame, columns):
    """Один целочисленный ключ на строку по значениям колонок; одинаковые значения - один ключ."""
    return pd.util.hash_pandas_object(frame[columns], index=False).to_numpy().view(np.int64)


def detect_entity_key(X_train, X_test, exclude=(), periods=None):
    """Ищет «сущность», которая повторяется в train и test: здание, клиента, устройство.

    Жадно собирает ключ из колонок, начиная с самых разнообразных. Две меры ключа:
    - повторяемость: доля строк train, чей ключ встречается в нескольких периодах
      (periods - колонка периода при групповой валидации) или просто в нескольких строках;
    - покрытие: доля строк test, чей ключ встречался в train.
    Колонка входит в ключ, если дробит группы (разводит разные сущности с одинаковой
    площадью) и почти не снижает обе меры. Признак самой сущности (год постройки, тип)
    делит ее историю целиком; то, что меняется по периодам (погода, текущие показатели),
    режет историю на куски, и повторяемость падает - такие колонки в ключ не идут.
    Возвращает {"columns", "groups", "repeat", "coverage"} или None, если повторов нет.
    """
    n = len(X_train)
    full = pd.concat([X_train, X_test], ignore_index=True)
    nunique = {c: X_train[c].nunique() for c in X_train if c in X_test and c not in exclude}
    order = sorted((c for c, u in nunique.items() if 2 <= u <= 0.9 * n), key=lambda c: -nunique[c])
    periods = None if periods is None else pd.Series(np.asarray(periods))

    def stats(columns):
        keys = _hash_rows(full, columns)
        train_keys = pd.Series(keys[:n])
        if periods is not None:
            repeated = periods.groupby(train_keys).transform("nunique") >= 2
        else:
            repeated = train_keys.map(train_keys.value_counts()) >= 2
        seen = train_keys.unique()
        return len(seen), float(repeated.mean()), float(np.isin(keys[n:], seen).mean())

    key, groups, repeat, coverage = [], 1, 1.0, 1.0
    for column in order[:ENTITY_CANDIDATES]:
        g, rep, cov = stats(key + [column])
        if not key:
            ok = rep >= 0.5 and cov >= 0.5
        else:
            ok = (g > 1.001 * groups and rep >= ENTITY_KEEP_REPEAT * repeat
                  and cov >= ENTITY_KEEP_COVERAGE * coverage)
        if ok:
            key, groups, repeat, coverage = key + [column], g, rep, cov
        if len(key) >= ENTITY_MAX_COLUMNS:
            break
    # Сущность имеет смысл, только если она повторяется и встречается в test
    if not key or coverage < 0.3 or repeat < 0.3 or groups < 20 or n / groups < 1.5:
        return None
    return {"columns": key, "groups": groups, "repeat": round(repeat, 4), "coverage": round(coverage, 4)}


DUPLICATE_MIN_COVERAGE = 0.05


def _exact_row_keys(frame, columns):
    # Factorize the actual tuples, including missing values. Equal numeric
    # values (for example +0.0 and -0.0) must identify the same row.
    return frame.groupby(columns, sort=False, observed=True, dropna=False).ngroup().to_numpy()


def duplicate_coverage(X_train, X_test, columns):
    """Доля строк test, у которых в train есть строка с точно такими же значениями columns."""
    full = pd.concat([X_train[columns], X_test[columns]], ignore_index=True)
    keys = _exact_row_keys(full, columns)
    return float(np.isin(keys[len(X_train):], keys[:len(X_train)]).mean())


def target_encoding_keys(X_train, X_test, exclude=(), periods=None):
    """Ключи для кодирования по таргету → {имя: список колонок}, пустой, если кодировать нечего.

    Если у заметной доли строк test есть точный дубликат в train (одинаковые записи трафика,
    повторные заявки), ключ - все признаки: модель узнает, какие метки бывают у такой же
    строки. Ключ детерминированный, в отличие от найденной сущности, и на UNSW дал на
    отложенных строках Macro F1 0.6322 -> 0.6477 (сущность из 6 колонок: 0.6393 или хуже базы).
    Иначе ищем сущность, которая повторяется по периодам (здание в разные годы).
    Отдельные категории не кодируем: LightGBM и так хорошо с ними работает.
    """
    columns = [c for c in X_train if c in X_test]
    if columns and len(X_test) and duplicate_coverage(X_train, X_test, columns) >= DUPLICATE_MIN_COVERAGE:
        return {"duplicate": columns}
    entity = detect_entity_key(X_train, X_test, exclude, periods)
    return {"entity": entity["columns"]} if entity else {}


def add_key_columns(X_train, X_test, specs, *, raw_frames=None):
    """Добавляет колонки-ключи TE_PREFIX + имя. Модель их не видит, только их кодировки."""
    full = pd.concat(raw_frames if raw_frames is not None else [X_train, X_test], ignore_index=True)
    n = len(X_train)
    X_train, X_test = X_train.copy(), X_test.copy()
    names = []
    for name, columns in specs.items():
        keys = _exact_row_keys(full, columns) if name == "duplicate" else _hash_rows(full, columns)
        column = TE_PREFIX + str(name)
        X_train[column], X_test[column] = keys[:n], keys[n:]
        names.append(column)
    return X_train, X_test, names
