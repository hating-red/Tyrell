"""Автоматические признаки без LLM.

Простые преобразования, полезные почти на любой таблице: части дат, частоты категорий,
число пропусков в строке, средние и отклонения важных числовых признаков по категориям.
Таргет не используется, train и test обрабатываются одной таблицей, поэтому утечки нет.
Полезность набора проверяется снаружи на тех же фолдах, что и остальные эксперименты.
"""

import numpy as np
import pandas as pd

MAX_GROUP_COLUMNS = 3             # сколько категорий берем для групповых признаков
MAX_GROUP_NUMERIC = 5             # сколько числовых признаков агрегируем по группам
MAX_GROUP_CARDINALITY = 1000      # категории с большим числом значений для групп не годятся
DATE_PATTERN = r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}"


def _is_categorical(s):
    return (isinstance(s.dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(s)
            or pd.api.types.is_string_dtype(s))


def _is_numeric(s):
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def _parse_dates(s):
    """Строковая колонка с датами в datetime, иначе None."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    if not _is_categorical(s):
        return None
    sample = s.dropna().astype(str).head(500)
    if sample.empty or sample.str.contains(DATE_PATTERN).mean() < 0.9:
        return None
    dates = pd.to_datetime(s, errors="coerce")
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


def add_auto_features(X_train, X_test, numeric_priority=None):
    """Возвращает X_train и X_test с новыми колонками; исходные колонки не меняются."""
    full = pd.concat([X_train, X_test], ignore_index=True)
    new = {}

    categorical = [c for c in full if _is_categorical(full[c])]
    for column in list(categorical):
        dates = _parse_dates(full[column])
        if dates is None:
            continue
        categorical.remove(column)
        new[f"{column}__year"] = dates.dt.year
        new[f"{column}__month"] = dates.dt.month
        new[f"{column}__dayofweek"] = dates.dt.dayofweek
        new[f"{column}__dayofyear"] = dates.dt.dayofyear
        if (dates.dt.hour.fillna(0) != 0).any():
            new[f"{column}__hour"] = dates.dt.hour

    for column in categorical:
        counts = full[column].value_counts()
        new[f"{column}__freq"] = full[column].map(counts).astype(float)

    missing = full.isna().sum(axis=1)
    if missing.any():
        new["__missing_count"] = missing.astype(float)

    groups = [c for c in categorical if 2 <= X_train[c].nunique() <= MAX_GROUP_CARDINALITY]
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
