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


def _hash_rows(frame, columns):
    """Один целочисленный ключ на строку по значениям колонок; одинаковые значения - один ключ."""
    return pd.util.hash_pandas_object(frame[columns].astype(str), index=False).to_numpy().view(np.int64)


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


def target_encoding_keys(X_train, X_test, exclude=(), periods=None):
    """Ключи для кодирования по таргету → {имя: список колонок}, пустой, если сущности нет.

    Только сущность: на публичной регрессии кодирование отдельной категории ничего не дало,
    LightGBM и так хорошо работает с категориями. Отдельные колонки агент может задать сам.
    """
    entity = detect_entity_key(X_train, X_test, exclude, periods)
    return {"entity": entity["columns"]} if entity else {}


def add_key_columns(X_train, X_test, specs):
    """Добавляет колонки-ключи TE_PREFIX + имя. Модель их не видит, только их кодировки."""
    full = pd.concat([X_train, X_test], ignore_index=True)
    n = len(X_train)
    X_train, X_test = X_train.copy(), X_test.copy()
    names = []
    for name, columns in specs.items():
        keys = _hash_rows(full, columns)
        column = TE_PREFIX + str(name)
        X_train[column], X_test[column] = keys[:n], keys[n:]
        names.append(column)
    return X_train, X_test, names
