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


def event_entities(fit, clock, *, maximum=4):
    """Identify repeated entity keys from the first historical fit prefix."""
    candidates = []
    for name in fit:
        if name == clock or not name.lower().endswith(("_id", "_key", "_code")):
            continue
        values = fit[name].dropna()
        unique = values.nunique()
        if 2 <= unique <= len(values) // 2:
            candidates.append((name, unique))
    # Preserve a mix of granular and coarse keys while cutting duplicate FK aliases.
    chosen = []
    for name, _ in sorted(candidates, key=lambda x: (-x[1], x[0])):
        if not any(fit[name].equals(fit[other]) for other in chosen):
            chosen.append(name)
        if len(chosen) >= maximum:
            break
    return chosen


def _codes(source, query, keys):
    if not keys:
        return np.zeros(len(source), dtype="int64"), np.zeros(len(query), dtype="int64")
    columns = pd.concat([source[list(keys)], query[list(keys)]], ignore_index=True)
    codes, _ = pd.factorize(pd.MultiIndex.from_frame(columns), sort=False)
    codes = np.asarray(codes, dtype="int64")
    codes[~columns.notna().all(axis=1).to_numpy()] = -1
    return codes[:len(source)], codes[len(source):]


def event_activity(source, query, clock, entities):
    """Use only earlier registrations, which can include unlabelled test events."""
    found = {}
    ts, qt = source[clock].to_numpy(), query[clock].to_numpy()
    for cols in [(), *((col,) for col in entities)]:
        past, now = _codes(source, query, cols)
        history = pd.DataFrame({"key": past, "stamp": ts})
        by_key = history.loc[history["stamp"].notna() & history.key.ge(0)].groupby("key").indices
        present = history.loc[history["stamp"].notna() & history.key.ge(0), "stamp"].to_numpy()
        title = "__activity_" + ("all" if not cols else cols[0])
        values = np.full((len(query), 6), np.nan)
        for key, ids in pd.Series(np.arange(len(query))).groupby(now, sort=False):
            pos = ids.to_numpy()
            pos = pos[pd.notna(qt[pos]) & (key >= 0)]
            if not len(pos):
                continue
            stamp = qt[pos]
            ordered = np.sort(present[by_key[key]]) if key in by_key else np.empty(0, dtype=ts.dtype)
            prev = np.searchsorted(ordered, stamp, side="left")
            for j, hours in enumerate((1, 3, 24, 168)):
                values[pos, j] = prev - np.searchsorted(ordered, stamp - np.timedelta64(hours, "h"))
            hit = prev > 0
            values[pos[hit], 4] = (stamp[hit] - ordered[prev[hit] - 1]) / np.timedelta64(1, "s")
            values[pos, 5] = 1 + prev - np.searchsorted(ordered, stamp.astype("datetime64[D]"))
        for j, name in enumerate(("1h", "3h", "24h", "7d", "seconds_since_previous", "day_rank")):
            found[f"{title}_{name}"] = values[:, j]
    return pd.DataFrame(found, index=query.index, dtype="float32")


def activity_reference(source, query, clock, entities):
    """Get each sampled row's window counts from cumulative earlier registrations."""
    fields = ("1h", "3h", "24h", "7d", "seconds_since_previous", "day_rank")
    outputs = {}
    for keys in [(), *((column,) for column in entities)]:
        past, now = _codes(source, query, keys)
        history = pd.DataFrame({"key": past, "stamp": source[clock].array})
        history = history.loc[history["stamp"].notna() & history["key"].ge(0)].sort_values("stamp")
        history["n"] = history.groupby("key", sort=False).cumcount() + 1
        ask = pd.DataFrame({"row": np.arange(len(query)), "key": now, "stamp": query[clock].array})
        ask = ask.loc[ask["stamp"].notna() & ask["key"].ge(0)].sort_values("stamp")

        def prefix(cutoff):
            markers = ask[["row", "key"]].assign(cutoff=cutoff)
            return pd.merge_asof(markers, history, by="key", left_on="cutoff", right_on="stamp",
                                 allow_exact_matches=False, direction="backward").set_index("row")

        previous = prefix(ask["stamp"])
        count = previous["n"].fillna(0)
        values = [count - prefix(ask["stamp"] - pd.Timedelta(hours=hours))["n"].fillna(0)
                  for hours in (1, 3, 24, 168)]
        values.extend([(previous["cutoff"] - previous["stamp"]).dt.total_seconds(),
                       1 + count - prefix(ask["stamp"].dt.floor("D"))["n"].fillna(0)])
        name = "__activity_" + (keys[0] if keys else "all")
        outputs.update({f"{name}_{field}": column.reindex(range(len(query))).to_numpy()
                        for field, column in zip(fields, values)})
    return pd.DataFrame(outputs, index=query.index, dtype="float32")


def causal_target_history(source, query, clock, available, labels, entities, *, window, smoothing, median=False):
    """Apply strictly earlier, observable labels from a fold's own model-fit prefix."""
    from itertools import combinations

    query_clock = query[clock].to_numpy()
    numeric = np.asarray(labels, dtype=float)
    avail = np.asarray(available)
    observed = pd.notna(avail) & np.isfinite(numeric)
    numeric, avail = numeric[observed], avail[observed]
    source = source.iloc[np.flatnonzero(observed)]
    order = np.argsort(avail, kind="stable")
    prefix = np.r_[0., np.cumsum(numeric[order])]
    prior_n = np.searchsorted(avail[order], query_clock, side="left")
    prior = np.divide(prefix[prior_n], prior_n, out=np.full(len(query), np.nan), where=prior_n > 0)
    outputs = {}
    ordered = pd.DataFrame({"event": query_clock, "row": np.arange(len(query))})
    ordered = ordered.dropna(subset=["event"]).sort_values("event", kind="stable")
    prior[pd.isna(query_clock)] = np.nan
    groups = [(c,) for c in entities] + list(combinations(entities[:3], 2))
    for cols in groups:
        source_key, query_key = _codes(source, query, cols)
        hist = pd.DataFrame({"key": source_key, "available": avail, "y": numeric})
        hist = hist.loc[hist.key.ge(0)].sort_values("available", kind="stable")
        g = hist.groupby("key", sort=False)
        running = g["y"].cumsum()
        hist["sum"] = running - running.groupby(hist["key"], sort=False).shift(window, fill_value=0.)
        hist["count"] = g.cumcount() + 1
        if median:
            hist["median"] = g["y"].rolling(window, min_periods=1).median().droplevel(0).reindex(hist.index)
        hist["last_positive"] = hist["available"].where(hist["y"].gt(0)).groupby(hist["key"], sort=False).ffill()
        ask = ordered.assign(key=query_key[ordered["row"].to_numpy()])
        hits = pd.merge_asof(ask, hist, by="key", left_on="event", right_on="available",
                             direction="backward", allow_exact_matches=False)
        if (hits["available"] >= hits["event"]).any():
            from leak_audit import LeakAuditError
            raise LeakAuditError("label was not available before the query event")
        hits = hits.set_index("row").reindex(range(len(query)))
        n = hits["count"].fillna(0).to_numpy()
        tag = "__history_" + "_by_".join(cols)
        outputs[tag + "_count"] = n
        outputs[tag + "_mean"] = (hits["sum"].fillna(0).to_numpy() + smoothing * prior) / (np.minimum(n, window) + smoothing)
        if median:
            outputs[tag + "_median"] = hits["median"].to_numpy()
        outputs[tag + "_since_positive"] = (hits["event"] - hits["last_positive"]).dt.total_seconds().to_numpy()
    return pd.DataFrame(outputs, index=query.index, dtype="float32")


def history_reference(source, query, clock, availability, labels, entities, *, window, smoothing, median,
                      corrupt_future=False):
    """Independently slice each entity's available prefix for every sampled query."""
    from itertools import combinations
    from leak_audit import replay_payload

    stamps = np.asarray(availability)
    labels = np.asarray(labels, dtype=float)
    observed = pd.notna(stamps) & np.isfinite(labels)
    order = np.flatnonzero(observed)
    order = order[np.argsort(stamps[order], kind="stable")]
    ask = query[clock].to_numpy()
    valid = pd.notna(ask)
    end = np.searchsorted(stamps[order], ask, side="left")
    # For each cutoff the perturbed array shares this prefix and has a constant
    # suffix. Its observable prefix sums are therefore shared across replays.
    totals = np.r_[0., np.cumsum(labels[order])]
    prior = np.divide(totals[end], end, out=np.full(len(query), np.nan), where=valid & (end > 0))
    outputs = {}
    for keys in [(c,) for c in entities] + list(combinations(entities[:3], 2)):
        src_keys, query_keys = _codes(source, query, keys)
        groups = pd.Series(order).groupby(src_keys[order], sort=False)
        past = {key: rows.to_numpy() for key, rows in groups if key >= 0}
        values = np.full((len(query), 4), np.nan)
        values[:, 0] = 0
        values[:, 1] = prior
        for key, positions in pd.Series(np.flatnonzero(valid)).groupby(query_keys[valid], sort=False):
            if key not in past:
                continue
            rows, ids = past[key], positions.to_numpy()
            times, ys = stamps[rows], labels[rows]
            cut = np.searchsorted(times, ask[ids], side="left")
            values[ids, 0] = cut
            positive = times[ys > 0]
            last = np.searchsorted(positive, ask[ids], side="left")
            hit = last > 0
            values[ids[hit], 3] = (ask[ids[hit]] - positive[last[hit]-1]) / np.timedelta64(1, "s")
            for idx, stop in zip(ids, cut):
                if stop:
                    start = max(0, stop-window)
                    recent = ys[start:stop]
                    if corrupt_future:
                        # Include the first noisy value, even for coincident labels.
                        # The rest of the identical suffix is outside this prefix.
                        tail = slice(start, min(len(ys), stop + 1))
                        changed = replay_payload(ys[tail], times[tail], ask[idx], corrupt_future=True)
                        recent = changed[np.isfinite(changed)]
                    values[idx, 1] = (recent.sum() + smoothing * prior[idx]) / (len(recent) + smoothing)
                    if median:
                        values[idx, 2] = np.median(recent)
        tag = "__history_" + "_by_".join(keys)
        columns = (("count", 0), ("mean", 1), ("median", 2), ("since_positive", 3))
        outputs.update({tag + "_" + col: values[:, i] for col, i in columns if median or col != "median"})
    return pd.DataFrame(outputs, index=query.index, dtype="float32")


def audit_event_kernel(builder, source, availability, queries, clock, group, *, labels=None, budget=None,
                       reference=None, corrupted_reference=None, train_rows=None):
    """Replay sampled queries after dropping future inputs and perturbing future labels."""
    from leak_audit import AuditBudget, check_equal, exact_cutoffs, sample_positions

    budget = budget or AuditBudget()
    groups = [queries] if train_rows is None else [queries.iloc[:train_rows], queries.iloc[train_rows:]]

    def verify(counts, check_time):
        queries = pd.concat([frame.iloc[sample_positions(len(frame), limit=count)]
                             for frame, count in zip(groups, counts)], ignore_index=True)
        expected = builder(source, queries, availability, labels)
        check_time()
        if reference is not None:
            check_equal(expected, reference(source, queries, availability, labels))
        if corrupted_reference is not None:
            check_time()
            check_equal(expected, corrupted_reference(source, queries, availability, labels))
        cutoffs = exact_cutoffs(queries[clock])
        if reference is not None:
            # A full input replay exercises the production entry point in addition
            # to the indexed replay, which covers every sampled cutoff.
            cutoffs = (next(cutoffs, None),)
        for item in cutoffs:
            if item is None:
                continue
            check_time()
            idx, cutoff = item
            original = queries.iloc[idx]
            retained = np.asarray(availability < cutoff)
            masked = builder(source.iloc[retained], original, np.asarray(availability)[retained],
                             None if labels is None else np.asarray(labels)[retained])
            check_equal(expected.iloc[idx], masked)
            check_time()
            changed = source.copy()
            noisy = None
            if labels is not None:
                noisy = np.asarray(labels, dtype=float).copy()
                noisy[~retained] = -123456.789
            else:
                for column in changed.columns.difference([clock]):
                    changed[column] = changed[column].astype(object)
                    changed.loc[~retained, column] = "audit changed future entity"
            check_equal(expected.iloc[idx], builder(changed, original, availability, noisy))
        checks = ["truncated_source", "corrupted_source"]
        if reference is not None:
            checks.append("per_row_cutoff")
        if corrupted_reference is not None or reference is None:
            checks.append("per_cutoff_corrupted_source")
        return dict(features=list(expected), cutoffs=int(queries[clock].nunique()), checks=checks,
                    rule="strictly earlier registration" if labels is None else "fold labels available strictly before event")

    budget.sampled(group, [len(frame) for frame in groups], verify)


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
