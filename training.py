"""Универсальное обучение табличной модели с кросс-валидацией.

Главная точка входа - train_and_predict(X, y, X_test, task_type, metric, config, time_budget).
Модуль не знает ничего о конкретном датасете: тип задачи и метрика приходят снаружи,
эксперимент задается словарем config, поэтому функцию можно вызывать в цикле гипотез.
"""

from __future__ import annotations

import copy
import importlib
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable

from data_tools import label_available_at as _label_available_at

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    log_loss,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

TASK_TYPES = ("regression", "binary", "multiclass")
CLASSIFICATION = ("binary", "multiclass")


# ---------------------------------------------------------------------------
# Метрики
# ---------------------------------------------------------------------------

def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    # Быстрая матрица ошибок через bincount, в разы быстрее sklearn
    k = int(max(y_true.max(), y_pred.max())) + 1
    return np.bincount(y_true * k + y_pred, minlength=k * k).reshape(k, k)


def _per_class_f1(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tp = np.diag(cm).astype(float)
    denom = 2 * tp + (cm.sum(0) - tp) + (cm.sum(1) - tp)
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    # Как в sklearn: классы, которых нет ни в y_true, ни в y_pred, не учитываются
    return f1, denom > 0


def _macro_f1(y_true, y_pred):
    f1, present = _per_class_f1(_confusion(y_true, y_pred))
    return float(f1[present].mean())


def _weighted_f1(y_true, y_pred):
    cm = _confusion(y_true, y_pred)
    f1, _ = _per_class_f1(cm)
    support = cm.sum(1)
    return float((f1 * support).sum() / support.sum())


def _binary_f1(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom else 0.0


def _accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def _balanced_accuracy(y_true, y_pred):
    cm = _confusion(y_true, y_pred)
    support = cm.sum(1)
    mask = support > 0
    return float((np.diag(cm)[mask] / support[mask]).mean())


def _mcc(y_true, y_pred):
    cm = _confusion(y_true, y_pred).astype(float)
    t_sum, p_sum = cm.sum(1), cm.sum(0)
    n, correct = cm.sum(), np.trace(cm)
    cov_tp = correct * n - t_sum @ p_sum
    cov_pp = n * n - p_sum @ p_sum
    cov_tt = n * n - t_sum @ t_sum
    denom = np.sqrt(cov_tt * cov_pp)
    return float(cov_tp / denom) if denom > 0 else 0.0


def _roc_auc(y_true, proba):
    if proba.shape[1] == 2:
        return float(roc_auc_score(y_true, proba[:, 1]))
    return float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro",
                               labels=np.arange(proba.shape[1])))


def _gini(y_true, proba):
    return 2 * _roc_auc(y_true, proba) - 1


def _pr_auc(y_true, proba):
    return float(average_precision_score(y_true, proba[:, 1]))


def _logloss(y_true, proba):
    # Быстрее sklearn, результат тот же: нормировка строк и клиппинг на машинный eps
    eps = np.finfo(proba.dtype).eps
    p = proba[np.arange(len(y_true)), y_true] / proba.sum(1)
    return float(-np.mean(np.log(np.clip(p, eps, 1 - eps))))


def _rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _rmsle(y_true, y_pred):
    return float(np.sqrt(np.mean((np.log1p(np.clip(y_pred, 0, None)) - np.log1p(y_true)) ** 2)))


@dataclass(frozen=True)
class Metric:
    name: str
    func: Callable[[np.ndarray, np.ndarray], float]
    greater_is_better: bool
    needs_proba: bool
    task_types: tuple[str, ...]


METRICS: dict[str, Metric] = {m.name: m for m in [
    # Регрессия
    Metric("rmse", _rmse, False, False, ("regression",)),
    Metric("mse", lambda y, p: float(mean_squared_error(y, p)), False, False, ("regression",)),
    Metric("mae", lambda y, p: float(mean_absolute_error(y, p)), False, False, ("regression",)),
    Metric("r2", lambda y, p: float(r2_score(y, p)), True, False, ("regression",)),
    Metric("rmsle", _rmsle, False, False, ("regression",)),
    Metric("mape", lambda y, p: float(mean_absolute_percentage_error(y, p)), False, False, ("regression",)),
    # Классификация по меткам
    Metric("macro_f1", _macro_f1, True, False, CLASSIFICATION),
    Metric("weighted_f1", _weighted_f1, True, False, CLASSIFICATION),
    Metric("f1", _binary_f1, True, False, ("binary",)),
    Metric("accuracy", _accuracy, True, False, CLASSIFICATION),
    Metric("balanced_accuracy", _balanced_accuracy, True, False, CLASSIFICATION),
    Metric("mcc", _mcc, True, False, CLASSIFICATION),
    # Классификация по вероятностям
    Metric("roc_auc", _roc_auc, True, True, CLASSIFICATION),
    Metric("gini", _gini, True, True, ("binary",)),
    Metric("pr_auc", _pr_auc, True, True, ("binary",)),
    Metric("logloss", _logloss, False, True, CLASSIFICATION),
]}

_METRIC_ALIASES = {
    "root_mean_squared_error": "rmse", "mean_squared_error": "mse",
    "mean_absolute_error": "mae", "l1": "mae", "l2": "mse", "r2_score": "r2",
    "f1_macro": "macro_f1", "macro_f1_score": "macro_f1", "f1_weighted": "weighted_f1",
    "f1_score": "f1", "binary_f1": "f1", "acc": "accuracy",
    "auc": "roc_auc", "rocauc": "roc_auc", "roc_auc_ovr": "roc_auc",
    "average_precision": "pr_auc", "prauc": "pr_auc",
    "log_loss": "logloss", "cross_entropy": "logloss", "mlogloss": "logloss",
    "matthews_corrcoef": "mcc",
}

DEFAULT_METRIC = {"regression": "rmse", "binary": "roc_auc", "multiclass": "macro_f1"}


def get_metric(name: str | None, task_type: str) -> Metric:
    """Найти метрику по имени. Неизвестная или несовместимая - дефолт по типу задачи."""
    key = (name or "").strip().lower().replace("-", "_").replace(" ", "_")
    key = _METRIC_ALIASES.get(key, key)
    if key == "f1" and task_type == "multiclass":
        key = "macro_f1"
    metric = METRICS.get(key)
    if metric is None:
        logger.warning("Неизвестная метрика %r, беру дефолт %s", name, DEFAULT_METRIC[task_type])
        return METRICS[DEFAULT_METRIC[task_type]]
    if task_type not in metric.task_types:
        logger.warning("Метрика %s не подходит для %s, беру дефолт %s",
                       key, task_type, DEFAULT_METRIC[task_type])
        return METRICS[DEFAULT_METRIC[task_type]]
    return metric


def _is_better(new: float, old: float, greater: bool, eps: float = 1e-12) -> bool:
    return new > old + eps if greater else new < old - eps


# ---------------------------------------------------------------------------
# Конфиг
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict[str, Any] = {
    "model": "lightgbm",          # lightgbm, catboost, xgboost
    "params": {},                 # переопределение гиперпараметров модели
    "features": None,             # список признаков, None - все колонки
    "n_folds": 5,
    "seed": 42,                   # сид модели
    "fold_seed": None,            # сид разбиения на фолды, None - равен seed
    "clip_target": True,          # регрессия: клиппинг предсказаний в диапазон таргета train
    "num_boost_round": 3000,
    "early_stopping_rounds": 100,
    "log_target": None,           # log1p таргета в регрессии, None - авто (только для rmsle)
    "class_weight": None,         # None или "balanced"
    "postprocess": True,          # подбор весов классов под метрику по меткам
    "n_jobs": None,               # None - min(8, число CPU)
    "split": None,                # None - случайные фолды; {"type": "group", "column": c} - группы
    "cat_encoding": "native",     # native - категории как есть; frequency - доля значения в train
    "target_encoding": None,      # колонки-ключи X: среднее таргета по ключу считается внутри фолдов
    "te_lag": True,
    "categorical_ids": False,
    "entity_history": False,
    "ad_history": False,
    "ad_history_window": 20,
    "ad_history_smoothing": 20.,
    "audit_required": False,
    "event_activity": False,
    "causal_history": False,
    "history_window": 64,
    "history_smoothing": 24.,
    "recency_days": None,
    "mae_objective": "l1",
    "mae_shift": False,
    "profile_categories": False,
    "label_delay_unit": None,     # duration target from event time until the label is known
    "refit_full": False,          # дообучить модель на всем train и усреднить с моделями фолдов
}

_ROUND_KEYS = ("n_estimators", "num_iterations", "num_boost_round", "iterations", "num_rounds")
_MIN_ROUNDS = 20                  # меньше этого по бюджету времени не урезаем
_AUDITED_CAT_ONE_HOT = 32
_TE_SMOOTHING = 2.0               # сглаживание среднего по ключу к общему среднему, «виртуальных строк»
_TE_INNER_FOLDS = 5               # внутренние фолды для кодирования строк обучающей части
_REFIT_ROUNDS = 1.15              # на всем train раундов больше, чем в среднем на фолде: данных больше


def _resolve_config(config: dict | None) -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    for key, value in (config or {}).items():
        if key not in DEFAULT_CONFIG:
            logger.warning("Неизвестный ключ конфига %r проигнорирован", key)
            continue
        cfg[key] = copy.deepcopy(value)
    cfg["params"] = dict(cfg["params"] or {})
    # Число раундов могли передать внутри params под любым из привычных имен
    for key in _ROUND_KEYS:
        if key in cfg["params"]:
            cfg["num_boost_round"] = int(cfg["params"].pop(key))
    cfg["model"] = str(cfg["model"]).strip().lower()
    max_jobs = min(8, os.cpu_count() or 1)
    requested = int(cfg["n_jobs"] or max_jobs)
    cfg["n_jobs"] = min(max_jobs, requested) if requested > 0 else max_jobs
    for key in ("num_threads", "num_thread", "nthread", "nthreads", "n_jobs", "thread_count"):
        cfg["params"].pop(key, None)
    for key in ("device", "devices", "device_type", "task_type", "gpu_id", "predictor"):
        cfg["params"].pop(key, None)
    if cfg["params"].get("tree_method") == "gpu_hist":
        cfg["params"]["tree_method"] = "hist"
    if cfg["fold_seed"] is None:
        cfg["fold_seed"] = cfg["seed"]
    return cfg


def _check_backend(name: str) -> str:
    aliases = {"lightgbm": "lightgbm", "lgbm": "lightgbm", "lgb": "lightgbm",
               "catboost": "catboost", "cb": "catboost", "xgboost": "xgboost", "xgb": "xgboost"}
    backend = aliases.get(name)
    if backend is None:
        logger.warning("Неизвестная модель %r, откат на LightGBM", name)
        return "lightgbm"
    if backend != "lightgbm":
        try:
            importlib.import_module(backend)
        except ImportError:
            logger.warning("Библиотека %s недоступна, откат на LightGBM", backend)
            return "lightgbm"
    return backend


def backend_available(name):
    try:
        importlib.import_module(name)
    except (ImportError, OSError):
        return False
    return True


# ---------------------------------------------------------------------------
# Подготовка данных
# ---------------------------------------------------------------------------

def _is_categorical(s: pd.Series) -> bool:
    return (isinstance(s.dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(s)
            or pd.api.types.is_string_dtype(s))


def unseen_category_share(X: pd.DataFrame, X_test: pd.DataFrame) -> dict[str, float]:
    """Доля строк test с категорией, которой нет в train, по каждой категориальной колонке."""
    shares = {}
    for col in X.columns:
        if col in X_test.columns and _is_categorical(X[col]):
            t = X_test[col]
            shares[col] = float((t.notna() & ~t.isin(set(X[col].dropna()))).mean())
    return shares


def _prepare_features(X: pd.DataFrame, X_test: pd.DataFrame, features: list | None,
                      cat_encoding: str = "native") -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Отбор признаков и приведение типов. Одна копия на датафрейм, исходники не меняются.

    cat_encoding="frequency" заменяет категорию ее долей в train, новые значения test получают 0.
    Это устойчивее к категориям, которых не было в обучении: они ведут себя как самые редкие.
    """
    if features is None:
        features = list(X.columns)
    missing = [c for c in features if c not in X.columns or c not in X_test.columns]
    if missing:
        raise ValueError(f"Признаки отсутствуют в X или X_test: {missing[:20]}")
    if not features:
        raise ValueError("Пустой список признаков")

    Xp, Xtp = X[features].copy(), X_test[features].copy()
    cat_cols = []
    for col in features:
        s = Xp[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            # Даты в секунды, NaT превращается в NaN
            for df in (Xp, Xtp):
                from data_tools import to_utc_naive
                d = to_utc_naive(df[col])
                df[col] = np.where(d.isna(), np.nan, d.astype("int64") / 1e9)
        elif _is_categorical(s) and cat_encoding == "frequency":
            freq = s.value_counts(normalize=True)
            Xp[col] = s.map(freq).astype(float)
            t = Xtp[col]
            Xtp[col] = t.map(freq).astype(float).where(t.isna() | t.isin(freq.index), 0.0)
        elif _is_categorical(s):
            cat = Xp[col].astype("category").cat.remove_unused_categories()
            Xp[col] = cat
            # Категории test строго по train, новые значения становятся NaN
            Xtp[col] = pd.Categorical(Xtp[col], categories=cat.cat.categories)
            cat_cols.append(col)
        elif pd.api.types.is_bool_dtype(s):
            Xp[col] = s.astype("float32")
            Xtp[col] = Xtp[col].astype("float32")
    return Xp, Xtp, cat_cols


def _to_catboost_frame(df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    # CatBoost требует строковые категории без NaN
    if not cat_cols:
        return df
    out = df.copy(deep=False)
    for col in cat_cols:
        out[col] = df[col].astype(object).where(df[col].notna(), "__nan__").astype(str)
    return out


def _time_values(series):
    if pd.api.types.is_numeric_dtype(series):
        return series.replace([np.inf, -np.inf], np.nan)
    from data_tools import parse_event_datetime
    return parse_event_datetime(series)


def _time_folds(times, y_enc, n_folds, task_type, metric):
    """Expanding windows of distinct event times; equal timestamps stay together."""
    ranks, unique = pd.factorize(times, sort=True)
    if len(unique) < 2:
        raise ValueError("Chronological validation needs at least two distinct event times")
    blocks = np.array_split(np.arange(len(unique)), min(len(unique), max(2, int(n_folds) + 1)))
    folds = []
    n_classes = len(np.unique(y_enc))
    for block in blocks[1:]:
        train = np.flatnonzero((ranks >= 0) & (ranks < block[0]))
        valid = np.flatnonzero((ranks >= block[0]) & (ranks <= block[-1]))
        if task_type in CLASSIFICATION and np.unique(y_enc[train]).size < n_classes:
            continue
        if metric in ("roc_auc", "gini") and np.unique(y_enc[valid]).size < n_classes:
            continue
        folds.append((train, valid))
    if not folds:
        raise ValueError("No chronological validation window has sufficient class coverage")
    return folds


def _past_monitor(train_idx, times, y_fit, task_type, available_at=None):
    """Use a chronological training tail for early stopping without reading outer labels."""
    ranks, distinct = pd.factorize(times.iloc[train_idx], sort=True)
    cut = len(distinct) - max(1, (len(distinct) + 9) // 10)
    prefix, tail = train_idx[ranks < cut], train_idx[ranks >= cut]
    if available_at is not None and len(tail):
        prefix = prefix[(available_at.iloc[prefix] < times.iloc[tail].min()).to_numpy()]
    if len(prefix) >= 2 and (task_type not in CLASSIFICATION or
                             np.unique(y_fit[prefix]).size == np.unique(y_fit[train_idx]).size):
        return prefix, tail, "past_holdout"
    # A single time group or a class first seen in the tail prevents an inner split.
    # Bound the rounds and use a training-only monitor; no outer labels reach the backend.
    return train_idx, train_idx[:2048], "training_monitor"


def _make_folds(y_enc: np.ndarray, task_type: str, n_folds: int, seed: int,
                groups: np.ndarray | None = None) -> list:
    n = len(y_enc)
    n_folds = max(2, min(int(n_folds), n))
    if groups is not None:
        # Группа целиком в одном фолде: валидация похожа на test с новыми значениями группы
        n_groups = len(pd.unique(groups))
        if n_groups >= 2:
            n_folds = min(n_folds, n_groups)
            if task_type in CLASSIFICATION:
                splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
                return list(splitter.split(np.zeros(n), y_enc, groups))
            return list(GroupKFold(n_splits=n_folds).split(np.zeros(n), groups=groups))
        logger.warning("В колонке разбиения меньше двух групп, использую случайные фолды")
    if task_type in CLASSIFICATION:
        min_count = int(np.bincount(y_enc).min())
        if min_count < n_folds:
            new = max(2, min_count)
            logger.warning("В редком классе %d объектов, число фолдов %d -> %d", min_count, n_folds, new)
            n_folds = new
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        return list(splitter.split(np.zeros(n), y_enc))
    splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(n)))


# ---------------------------------------------------------------------------
# Кодирование по таргету
# ---------------------------------------------------------------------------

def _te_targets(y_scaled: np.ndarray, y_enc: np.ndarray, task_type: str, n_classes: int) -> np.ndarray:
    """Что усредняем по ключу: таргет регрессии, долю класса 1 или доли всех классов."""
    if task_type == "regression":
        return y_scaled.reshape(-1, 1).astype(float)
    if task_type == "binary":
        return (y_enc == 1).astype(float).reshape(-1, 1)
    return np.eye(n_classes)[y_enc]


def _te_stats(keys_fit: np.ndarray, Y_fit: np.ndarray, keys_apply: np.ndarray) -> np.ndarray:
    """Сглаженное среднее Y по ключу и число строк с ключом. Новый ключ -> NaN и 0."""
    prior = Y_fit.mean(axis=0)
    frame = pd.DataFrame(Y_fit)
    frame["__key"] = keys_fit
    grouped = frame.groupby("__key", sort=False)
    sums, counts = grouped.sum(), grouped.size()
    pos = sums.index.get_indexer(keys_apply)
    seen = pos >= 0
    out = np.full((len(keys_apply), Y_fit.shape[1] + 1), np.nan)
    n = counts.to_numpy()[pos[seen]].astype(float)
    out[seen, :-1] = (sums.to_numpy()[pos[seen]] + _TE_SMOOTHING * prior) / (n[:, None] + _TE_SMOOTHING)
    out[:, -1] = 0.0
    out[seen, -1] = n
    return out


def _te_lag(keys_fit: np.ndarray, periods_fit: np.ndarray, Y_fit: np.ndarray,
            keys_apply: np.ndarray, periods_apply: np.ndarray) -> np.ndarray:
    """Таргет той же сущности в ближайшем другом периоде и расстояние до него.

    Для нового года прошлогоднее значение сущности обычно точнее ее среднего за все годы:
    сущности со временем меняются. При равном расстоянии берем более ранний период.
    Свой период не используется. Нет других периодов -> NaN.
    """
    width = Y_fit.shape[1]
    fit = pd.DataFrame(Y_fit, columns=[f"y{j}" for j in range(width)])
    fit["key"], fit["period"] = keys_fit, periods_fit
    table = fit.groupby(["key", "period"], sort=False).mean().reset_index()
    apply = pd.DataFrame({"key": keys_apply, "period": periods_apply, "row": np.arange(len(keys_apply))})
    merged = apply.merge(table, on="key", suffixes=("", "_fit"))
    merged = merged[merged["period_fit"] != merged["period"]]
    distance = (merged["period_fit"] - merged["period"]).abs()
    merged = merged.assign(order=distance + 0.1 * (merged["period_fit"] > merged["period"]), distance=distance)
    best = merged.sort_values(["row", "order"]).drop_duplicates("row")
    out = np.full((len(keys_apply), width + 1), np.nan)
    rows = best["row"].to_numpy()
    out[rows, :width] = best[[f"y{j}" for j in range(width)]].to_numpy()
    out[rows, width] = best["distance"].to_numpy()
    return out


def _target_encode(keys: dict, keys_test: dict, Y: np.ndarray, tr_idx: np.ndarray, va_idx: np.ndarray,
                   groups: np.ndarray | None, seed: int,
                   periods: tuple | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Признаки по таргету для одного фолда: (обучающая часть, валидация, test).

    Валидация и test кодируются статистикой всей обучающей части. Строки самой обучающей
    части - внутренними фолдами, чтобы строка не видела свой таргет: иначе модель решит,
    что признак точнее, чем он будет на test. Внутренние фолды повторяют внешнее разбиение
    (группами, если оно групповое), чтобы признак вел себя так же, как на test.
    periods - (периоды размеченных строк, периоды test): тогда добавляется значение
    сущности в ближайшем другом периоде.
    """
    inner_groups = None if groups is None else groups[tr_idx]
    if inner_groups is not None and len(pd.unique(inner_groups)) >= 2:
        n_inner = min(_TE_INNER_FOLDS, len(pd.unique(inner_groups)))
        inner = list(GroupKFold(n_splits=n_inner).split(tr_idx, groups=inner_groups))
    else:
        inner = list(KFold(n_splits=min(_TE_INNER_FOLDS, len(tr_idx)), shuffle=True,
                           random_state=seed).split(tr_idx))
    parts_tr, parts_va, parts_te = [], [], []
    for name in keys:
        k_tr, Y_tr = keys[name][tr_idx], Y[tr_idx]
        enc_tr = np.full((len(tr_idx), Y.shape[1] + 1), np.nan)
        for fit, apply in inner:
            enc_tr[apply] = _te_stats(k_tr[fit], Y_tr[fit], k_tr[apply])
        parts_tr.append(enc_tr)
        parts_va.append(_te_stats(k_tr, Y_tr, keys[name][va_idx]))
        parts_te.append(_te_stats(k_tr, Y_tr, keys_test[name]))
        if periods is not None:
            p_lab, p_test = periods
            p_tr = p_lab[tr_idx]
            lag_tr = np.full((len(tr_idx), Y.shape[1] + 1), np.nan)
            for fit, apply in inner:
                lag_tr[apply] = _te_lag(k_tr[fit], p_tr[fit], Y_tr[fit], k_tr[apply], p_tr[apply])
            parts_tr.append(lag_tr)
            parts_va.append(_te_lag(k_tr, p_tr, Y_tr, keys[name][va_idx], p_lab[va_idx]))
            parts_te.append(_te_lag(k_tr, p_tr, Y_tr, keys_test[name], p_test))
    return np.hstack(parts_tr), np.hstack(parts_va), np.hstack(parts_te)


def _te_columns(names: list[str], width: int, lag: bool = False) -> list[str]:
    columns = []
    for name in names:
        columns += [f"{name}__te{j}" for j in range(width - 1)] + [f"{name}__te_count"]
        if lag:
            columns += [f"{name}__lag{j}" for j in range(width - 1)] + [f"{name}__lag_distance"]
    return columns


def _with_columns(X: pd.DataFrame, values: np.ndarray, columns: list[str]) -> pd.DataFrame:
    return pd.concat([X, pd.DataFrame(values, index=X.index, columns=columns)], axis=1)


def _past_encoding(fit_keys, available, targets, query_keys, query_times, smoothing=0):
    """Encode each row with labels that were already observable for its key."""
    history = pd.DataFrame(targets)
    values = list(history.columns)
    # Factorize once so pandas receives matching int64 key buffers on every platform.
    codes, _ = pd.factorize(np.concatenate([fit_keys, query_keys]))
    history["key"] = codes[:len(fit_keys)].astype("int64")
    history["when"] = available.array
    history = history.sort_values("when", kind="stable")
    grouped = history.groupby("key", sort=False)
    history["count"] = grouped.cumcount() + 1
    history[values] = grouped[values].cumsum()
    query = pd.DataFrame({"key": codes[len(fit_keys):].astype("int64"), "when": query_times.array})
    ordered = query.dropna(subset=["when"]).sort_values("when", kind="stable")
    prior = pd.merge_asof(ordered, history, on="when", by="key", allow_exact_matches=False)
    prior.index = ordered.index
    prior = prior.reindex(query.index)
    counts = prior["count"].fillna(0).to_numpy()
    if smoothing:
        pos = history["when"].searchsorted(query_times, side="left")
        order = np.argsort(available.to_numpy(), kind="stable")
        totals = np.vstack([np.zeros(len(values)), targets[order].cumsum(axis=0)])
        global_mean = totals[pos] / np.maximum(pos, 1)[:, None]
        mean = (prior[values].fillna(0).to_numpy() + smoothing * global_mean) / (counts[:, None] + smoothing)
        mean[(pos == 0) | query_times.isna().to_numpy()] = np.nan
    else:
        mean = prior[values].div(prior["count"], axis=0)
    return np.column_stack([mean, np.log1p(counts)])


# ---------------------------------------------------------------------------
# Бэкенды моделей
# ---------------------------------------------------------------------------

def _lgb_params(task_type, metric, n_classes, cfg):
    params = {
        "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 20,
        "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
        "lambda_l2": 1.0, "max_bin": 255, "verbosity": -1,
        "num_threads": cfg["n_jobs"], "seed": cfg["seed"],
    }
    if task_type == "regression":
        objective = {"mae": "l1", "mape": "mape"}.get(metric.name, "regression")
        if metric.name == "mae":
            objective = cfg.get("mae_objective", "l1")
            if objective == "quantile":
                params["alpha"] = .5
        params.update(objective=objective, metric="l1" if metric.name == "mae" else "l2")
    elif task_type == "binary":
        params.update(objective="binary", metric="average_precision" if metric.name == "pr_auc" else "binary_logloss")
    else:
        params.update(objective="multiclass", num_class=n_classes, metric="multi_logloss")
    params.update(cfg["params"])
    return params


def _fit_lightgbm(Xtr, ytr, wtr, Xva, yva, Xte, task_type, metric, n_classes, cfg, deadline):
    import lightgbm as lgb

    params = _lgb_params(task_type, metric, n_classes, cfg)
    # A single-class monitor cannot rank positives against negatives.
    if task_type == "binary" and params.get("metric") == "average_precision" and np.unique(yva).size < 2:
        params["metric"] = "binary_logloss"
    dtrain = lgb.Dataset(Xtr, ytr, weight=wtr)
    dvalid = lgb.Dataset(Xva, yva, reference=dtrain)
    state = {"time_stopped": False, "best_iter": 0, "best": None}

    def time_callback(env):
        # Сами отслеживаем лучшую итерацию, чтобы при остановке по времени не потерять ее
        _, _, value, higher = env.evaluation_result_list[0][:4]
        if state["best"] is None or _is_better(value, state["best"], higher):
            state["best"], state["best_iter"] = value, env.iteration
        if time.monotonic() >= deadline:
            state["time_stopped"] = True
            raise lgb.callback.EarlyStopException(state["best_iter"], env.evaluation_result_list)
    time_callback.order = 40

    evaluation = {}
    booster = lgb.train(
        params, dtrain, num_boost_round=cfg["num_boost_round"], valid_sets=[dvalid],
        callbacks=[lgb.record_evaluation(evaluation), time_callback] + ([] if cfg.get("final_fit") else
                   [lgb.early_stopping(cfg["early_stopping_rounds"], verbose=False)]),
    )
    best = booster.current_iteration() if cfg.get("final_fit") else booster.best_iteration or booster.current_iteration()
    pva = booster.predict(Xva, num_iteration=best, num_threads=cfg["n_jobs"])
    pte = booster.predict(Xte, num_iteration=best, num_threads=cfg["n_jobs"])
    if task_type == "binary":
        pva, pte = np.column_stack([1 - pva, pva]), np.column_stack([1 - pte, pte])
    monitor_history = next(iter(evaluation.values()))
    last = len(next(iter(monitor_history.values())))
    return pva, pte, best, state["time_stopped"], last


def _fit_xgboost(Xtr, ytr, wtr, Xva, yva, Xte, task_type, metric, n_classes, cfg, deadline):
    import xgboost as xgb

    params = {
        "eta": 0.05, "max_depth": 7, "subsample": 0.8, "colsample_bytree": 0.8,
        "min_child_weight": 1.0, "lambda": 1.0, "tree_method": "hist", "max_bin": 256,
        "nthread": cfg["n_jobs"], "seed": cfg["seed"], "verbosity": 0,
    }
    if task_type == "regression":
        if metric.name == "mae":
            objective = cfg.get("mae_objective", "l1")
            params.update(objective={"l1": "reg:absoluteerror", "quantile": "reg:quantileerror",
                                     "huber": "reg:pseudohubererror"}[objective], eval_metric="mae")
            if objective == "quantile":
                params["quantile_alpha"] = .5
        else:
            params.update(objective="reg:squarederror", eval_metric="rmse")
    elif task_type == "binary":
        params.update(objective="binary:logistic", eval_metric="logloss")
    else:
        params.update(objective="multi:softprob", num_class=n_classes, eval_metric="mlogloss")
    params.update(cfg["params"])

    class TimeCallback(xgb.callback.TrainingCallback):
        stopped = False

        def after_iteration(self, model, epoch, evals_log):
            if time.monotonic() >= deadline:
                self.stopped = True
                return True
            return False

    timer = TimeCallback()
    dtrain = xgb.DMatrix(Xtr, ytr, weight=wtr, enable_categorical=True, nthread=cfg["n_jobs"])
    dvalid = xgb.DMatrix(Xva, yva, enable_categorical=True, nthread=cfg["n_jobs"])
    monitor_ap = metric.name == "pr_auc" and np.unique(yva).size == 2
    def ap_metric(pred, dataset):
        return "average_precision", float(average_precision_score(dataset.get_label(), pred))
    booster = xgb.train(
        params, dtrain, num_boost_round=cfg["num_boost_round"], evals=[(dvalid, "valid")],
        custom_metric=ap_metric if monitor_ap else None,
        callbacks=[xgb.callback.EarlyStopping(cfg["early_stopping_rounds"], maximize=True if monitor_ap else None), timer],
        verbose_eval=False,
    )
    try:
        best = booster.best_iteration + 1
    except AttributeError:
        best = booster.num_boosted_rounds()
    if cfg.get("final_fit"):
        best = booster.num_boosted_rounds()
    dtest = xgb.DMatrix(Xte, enable_categorical=True, nthread=cfg["n_jobs"])
    pva = booster.predict(dvalid, iteration_range=(0, best))
    pte = booster.predict(dtest, iteration_range=(0, best))
    if task_type == "binary":
        pva, pte = np.column_stack([1 - pva, pva]), np.column_stack([1 - pte, pte])
    return pva, pte, best, timer.stopped, booster.num_boosted_rounds()


class _CatAveragePrecision:
    def evaluate(self, approxes, target, weight):
        return float(average_precision_score(target, approxes[0])), 1.

    def get_final_error(self, error, weight):
        return error / weight

    def is_max_optimal(self):
        return True


def _fit_catboost(Xtr, ytr, wtr, Xva, yva, Xte, task_type, metric, n_classes, cfg, deadline,
                  cat_cols=()):
    from catboost import CatBoost, Pool

    params = {
        # one_hot_max_size: низкокардинальные категории в one-hot, CTR на них в разы медленнее
        "learning_rate": 0.08, "depth": 6, "l2_leaf_reg": 3.0, "one_hot_max_size": 10,
        "iterations": cfg["num_boost_round"], "early_stopping_rounds": cfg["early_stopping_rounds"],
        "thread_count": cfg["n_jobs"], "random_seed": cfg["seed"],
        "verbose": False, "allow_writing_files": False, "use_best_model": not cfg.get("final_fit", False),
    }
    if task_type == "regression":
        params["loss_function"] = "MAE" if metric.name == "mae" else "RMSE"
        if metric.name == "mae":
            params["loss_function"] = {"l1": "MAE", "quantile": "Quantile:alpha=0.5",
                                       "huber": "Huber:delta=1.0"}[cfg.get("mae_objective", "l1")]
            params["eval_metric"] = "MAE"
    elif task_type == "binary":
        params.update(loss_function="Logloss", class_names=[0, 1])
        if metric.name == "pr_auc" and np.unique(yva).size == 2:
            params["eval_metric"] = _CatAveragePrecision()
    else:
        params.update(loss_function="MultiClass", class_names=list(range(n_classes)))
    params.update(cfg["params"])
    if cfg.get("audit_required") and (cfg.get("split") or {}).get("type") == "time":
        params["one_hot_max_size"] = _AUDITED_CAT_ONE_HOT

    class TimeCallback:
        stopped = False

        last_round = 0

        def after_iteration(self, info):
            self.last_round = info.iteration
            if time.monotonic() >= deadline:
                self.stopped = True
                return False
            return True

    timer = TimeCallback()
    cats = [c for c in cat_cols if c in Xtr.columns]
    model = CatBoost(params)
    model.fit(Pool(Xtr, ytr, weight=wtr, cat_features=cats),
              eval_set=Pool(Xva, yva, cat_features=cats), callbacks=[timer])
    kind = "RawFormulaVal" if task_type == "regression" else "Probability"
    pva = model.predict(Pool(Xva, cat_features=cats), prediction_type=kind, thread_count=cfg["n_jobs"])
    pte = model.predict(Pool(Xte, cat_features=cats), prediction_type=kind, thread_count=cfg["n_jobs"])
    return np.asarray(pva), np.asarray(pte), int(model.get_best_iteration() or 0) + 1, timer.stopped, timer.last_round


_BACKENDS = {"lightgbm": _fit_lightgbm, "xgboost": _fit_xgboost, "catboost": _fit_catboost}


# ---------------------------------------------------------------------------
# Постобработка: веса классов под метрику по меткам
# ---------------------------------------------------------------------------

_WEIGHT_GRID = np.exp(np.linspace(-3, 3, 41))


def fit_class_weights(proba: np.ndarray, y: np.ndarray, metric: Metric, n_passes: int = 3,
                      max_rows: int = 200_000) -> tuple[np.ndarray, float]:
    """Покоординатный поиск множителей классов: pred = argmax(proba * w).

    Для бинарной задачи это эквивалент подбора порога. На больших данных поиск идет
    по случайной подвыборке из max_rows строк, чтобы время не росло с размером.
    """
    if len(y) > max_rows:
        idx = np.random.default_rng(0).choice(len(y), max_rows, replace=False)
        proba, y = proba[idx], y[idx]
    n, k = proba.shape
    rows = np.arange(n)
    w = np.ones(k)
    best = metric.func(y, proba.argmax(1))
    for _ in range(n_passes):
        improved = False
        for c in range(k):
            # При изменении только w[c] конкурент каждой строки фиксирован, считаем его один раз
            others = proba * w
            others[:, c] = -np.inf
            rival = others.argmax(1)
            rival_score = others[rows, rival]
            pc = proba[:, c]
            for g in _WEIGHT_GRID:
                s = metric.func(y, np.where(pc * g > rival_score, c, rival))
                if _is_better(s, best, metric.greater_is_better):
                    best, w[c], improved = s, g, True
        if not improved:
            break
    return w, best


def _apply_weights(proba: np.ndarray, w: np.ndarray) -> np.ndarray:
    return (proba * w).argmax(1)


def _cross_fit_class_weights(oof: np.ndarray, y_enc: np.ndarray, folds: list, metric: Metric,
                             score_raw: float) -> tuple[np.ndarray, list | None, dict]:
    """Честная оценка весов классов: для каждого фолда веса подбираются на остальных.

    folds - только выполненные фолды. Финальные веса подбираются на всех OOF.
    Если веса не улучшают скор, возвращаются единичные веса и None вместо скоров.
    """
    done_mask = np.zeros(len(y_enc), dtype=bool)
    for _, va_idx in folds:
        done_mask[va_idx] = True
    cross_scores = []
    for _, va_idx in folds:
        fit_mask = done_mask.copy()
        fit_mask[va_idx] = False
        w, _ = fit_class_weights(oof[fit_mask], y_enc[fit_mask], metric)
        cross_scores.append(metric.func(y_enc[va_idx], _apply_weights(oof[va_idx], w)))
    score_pp = float(np.mean(cross_scores))
    weights, _ = fit_class_weights(oof[done_mask], y_enc[done_mask], metric)
    accepted = _is_better(score_pp, score_raw, metric.greater_is_better)
    logger.info("Веса классов: %s %.5f -> %.5f (кросс-фиттинг), %s", metric.name,
                score_raw, score_pp, "применяю" if accepted else "не улучшает, отключаю")
    info = {"type": "class_weights", "applied": accepted, "score_after": score_pp,
            "weights_raw": weights}
    if not accepted:
        return np.ones(oof.shape[1]), None, info
    return weights, cross_scores, info


# ---------------------------------------------------------------------------
# Главная функция
# ---------------------------------------------------------------------------

def train_and_predict(X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame, task_type: str,
                      metric: str, config: dict | None = None,
                      time_budget: float | None = None) -> dict:
    """Кросс-валидация, OOF и предсказание на test средним по моделям фолдов.

    Строки X и y сопоставляются по позиции. Порядок строк X_test сохраняется.
    """
    t_start = time.monotonic()
    if time_budget is not None and (not np.isfinite(time_budget) or time_budget <= 0):
        raise ValueError("Model time budget must be positive and finite")
    deadline = t_start + time_budget if time_budget is not None else float("inf")

    task_type = str(task_type).strip().lower()
    if task_type not in TASK_TYPES:
        raise ValueError(f"task_type должен быть одним из {TASK_TYPES}, получено {task_type!r}")
    if len(X) != len(y):
        raise ValueError(f"Размеры X ({len(X)}) и y ({len(y)}) не совпадают")
    if len(X_test) == 0:
        raise ValueError("X_test пустой")

    cfg = _resolve_config(config)
    if "split" not in (config or {}):
        from data_tools import detect_time_column
        clock = detect_time_column(X, X_test)
        if clock is not None:
            cfg["split"] = {"type": "time", "column": clock}
    cfg["model"] = _check_backend(cfg["model"])

    # Строки без таргета не участвуют в обучении, в OOF у них будет NaN
    y_series = pd.Series(np.asarray(y))
    labeled = y_series.notna().to_numpy()
    split = cfg["split"] or {}
    temporal = split.get("type") == "time"
    if split and (split.get("type") not in ("time", "group") or split.get("column") not in X):
        raise ValueError("Validation split requires an existing input column and type time or group")
    times = _time_values(X[split["column"]]) if temporal else None
    if temporal:
        labeled = labeled & times.notna().to_numpy()
    if not labeled.all():
        logger.warning("Пропуски в таргете: %d строк исключены из обучения", int((~labeled).sum()))
    if labeled.sum() < 10:
        raise ValueError("Слишком мало строк с таргетом для обучения")
    y_lab = y_series[labeled]

    # Кодирование таргета
    encoder = None
    if task_type == "regression":
        try:
            y_fit = pd.to_numeric(y_lab, errors="raise").to_numpy(dtype=float)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Регрессия, но таргет не числовой: {e}") from e
        n_classes = 0
    else:
        encoder = LabelEncoder()
        y_fit = encoder.fit_transform(y_lab.to_numpy())
        n_classes = len(encoder.classes_)
        if n_classes < 2:
            raise ValueError("В таргете меньше двух классов")
        # При расхождении верим данным
        if task_type == "binary" and n_classes > 2:
            logger.warning("Задача объявлена binary, но классов %d, переключаюсь на multiclass", n_classes)
            task_type = "multiclass"
        elif task_type == "multiclass" and n_classes == 2:
            logger.warning("Задача объявлена multiclass, но классов 2, переключаюсь на binary")
            task_type = "binary"

    metric_obj = get_metric(metric, task_type)

    # log1p таргета
    log_target = cfg["log_target"]
    if log_target is None:
        log_target = task_type == "regression" and metric_obj.name == "rmsle"
    if log_target and task_type != "regression":
        logger.warning("log_target имеет смысл только для регрессии, отключаю")
        log_target = False
    if log_target and not temporal and y_fit.min() < 0:
        logger.warning("В таргете есть отрицательные значения, log1p отключен")
        log_target = False
    cfg["log_target"] = bool(log_target)
    # Signed log1p agrees with log1p for positive durations. Its domain is independent
    # of later validation targets and it has a well-defined inverse for negative values.
    y_train_scale = np.sign(y_fit) * np.log1p(np.abs(y_fit)) if log_target else y_fit

    # Колонки-ключи для кодирования по таргету сами в модель не идут, только их кодировки
    te_cols = [c for c in (cfg["target_encoding"] or []) if c in X.columns and c in X_test.columns]
    features = [c for c in (cfg["features"] or list(X.columns)) if c not in te_cols]
    if temporal:
        # In a time split even categorical vocabularies and frequencies are fitted per window.
        Xp, Xtp, cat_cols = X[features].copy(deep=False), X_test[features].copy(deep=False), []
        clock = split["column"]
        if clock in features and not pd.api.types.is_numeric_dtype(X[clock]):
            Xp[clock] = times
            Xtp[clock] = _time_values(X_test[clock])
    else:
        Xp, Xtp, cat_cols = _prepare_features(X, X_test, features, cfg["cat_encoding"])
    cfg["features"] = list(Xp.columns)
    cfg["target_encoding"] = te_cols or None
    te_keys = {c: X[c].to_numpy()[labeled] for c in te_cols}
    te_keys_test = {c: X_test[c].to_numpy() for c in te_cols}
    groups = None
    if split.get("type") == "group":
        groups = X[split["column"]].to_numpy()[labeled]
    if not labeled.all():
        Xp = Xp.iloc[np.flatnonzero(labeled)]
    if cfg["model"] == "catboost" and not temporal:
        Xp, Xtp = _to_catboost_frame(Xp, cat_cols), _to_catboost_frame(Xtp, cat_cols)

    def training_weights(indices):
        weights = None
        if cfg["class_weight"] == "balanced" and task_type in CLASSIFICATION:
            local = y_fit[indices]
            counts = np.bincount(local, minlength=n_classes)
            weights = len(local) / (n_classes * counts[local])
        if cfg["recency_days"] and temporal and pd.api.types.is_datetime64_any_dtype(labeled_times):
            when = labeled_times.iloc[indices]
            age = (when.max() - when).dt.total_seconds().to_numpy()
            decay = np.exp(np.maximum(-80., -age / (86400 * cfg["recency_days"])))
            decay /= decay.mean()
            weights = decay if weights is None else weights * decay
        return weights

    if cfg["class_weight"] not in (None, "balanced"):
        logger.warning("class_weight=%r не поддерживается, игнорирую", cfg["class_weight"])

    labeled_times = times[labeled].reset_index(drop=True) if temporal else None
    available_at = None
    test_cutoff = None
    refit_idx = np.arange(len(y_fit))
    cutoff_details = []
    if cfg["label_delay_unit"]:
        if not temporal or task_type != "regression":
            raise ValueError("A duration label requires chronological regression")
        available_at = _label_available_at(labeled_times, y_fit, cfg["label_delay_unit"])
    elif temporal and cfg["audit_required"]:
        available_at = labeled_times
    if available_at is not None:
        test_times = _time_values(X_test[split["column"]])
        if test_times.isna().any():
            raise ValueError("Missing test event time prevents a label-availability cutoff")
        test_cutoff = test_times.min()
        refit_idx = np.flatnonzero(available_at < test_cutoff)
        if len(refit_idx) < 2:
            raise ValueError("Insufficient labels observed before the prediction period")
    if temporal:
        folds = _time_folds(labeled_times, y_fit, cfg["n_folds"], task_type, metric_obj.name)
        if available_at is not None:
            eligible = []
            for train, valid in folds:
                cutoff = labeled_times.iloc[valid].min()
                tr = train[(available_at.iloc[train] < min(test_cutoff, cutoff)).to_numpy()]
                va = valid[(available_at.iloc[valid] < test_cutoff).to_numpy()]
                if len(va) and len(tr) < 2:
                    raise ValueError("Insufficient observed labels before a validation window")
                if len(va):
                    eligible.append((tr, va))
                    cutoff_details.append({"cutoff": cutoff, "train_pending": len(train) - len(tr),
                                           "validation_pending_at_prediction": len(valid) - len(va)})
            if not eligible:
                raise ValueError("No evaluable validation window before the prediction period")
            folds = eligible
    else:
        folds = _make_folds(y_fit, task_type, cfg["n_folds"], cfg["fold_seed"], groups)
    clip = task_type == "regression" and bool(cfg["clip_target"])
    cfg["n_folds"] = len(folds)
    fit_fn = _BACKENDS[cfg["model"]]
    extra = {"cat_cols": cat_cols} if cfg["model"] == "catboost" else {}
    te_Y = _te_targets(y_train_scale, y_fit, task_type, n_classes) if te_cols or cfg["entity_history"] else None
    te_periods = None
    if te_cols and cfg["te_lag"] and groups is not None and pd.api.types.is_numeric_dtype(X[split['column']]):
        te_periods = (groups.astype(float), X_test[split['column']].to_numpy().astype(float))
    te_names = _te_columns(te_cols, te_Y.shape[1] + 1, te_periods is not None) if te_cols else []

    candidates = {"entity_history": [], "categorical_ids": []}
    general_entities = []
    datetime_clock = False
    ad_groups = {}
    probe_columns = []
    if temporal and clock in Xp:
        from features import ad_history_keys, temporal_entity_candidates
        first_fit, _, _ = _past_monitor(folds[0][0], labeled_times, y_fit, task_type, available_at)
        initial = Xp.iloc[first_fit]
        candidates = temporal_entity_candidates(initial, clock)
        datetime_clock = pd.api.types.is_datetime64_any_dtype(initial[clock])
        if datetime_clock:
            from features import event_entities
            general_entities = event_entities(initial, clock)
        if task_type == "binary" and metric_obj.name == "pr_auc" and pd.api.types.is_datetime64_any_dtype(initial[clock]):
            ad_groups = ad_history_keys(initial.columns)
        if cfg["profile_categories"] and metric_obj.name == "pr_auc" and task_type == "binary":
            for column, values in initial.items():
                if column == clock:
                    continue
                if (column in candidates["entity_history"] or _is_categorical(values) or
                        (pd.api.types.is_integer_dtype(values) and 2 <= values.nunique() <= 32)):
                    if len(probe_columns) < 12 and 2 <= values.nunique() <= min(50_000, len(initial) // 2):
                        probe_columns.append(column)
    for name, selected in candidates.items():
        if cfg[name] and not selected:
            raise ValueError(f"{name}: no eligible repeated IDs in the initial temporal fit prefix")
    if cfg["categorical_ids"] and cfg["cat_encoding"] != "native":
        raise ValueError("categorical_ids requires native category encoding")
    if cfg["ad_history"] and not ad_groups:
        raise ValueError("Advertising histories require time-ordered PR-AUC data and user and ad keys")
    if (cfg["event_activity"] or cfg["causal_history"]) and not datetime_clock:
        raise ValueError("Point-in-time histories require a datetime event clock")
    if cfg["causal_history"] and (not general_entities or task_type == "multiclass"):
        raise ValueError("This task has no eligible keys or scalar target for supervised histories")
    from leak_audit import AuditBudget, audit_log, sample_positions
    from features import activity_reference, audit_event_kernel, causal_target_history, event_activity, history_reference

    budget = AuditBudget()
    if cfg["event_activity"] or cfg["causal_history"]:
        train_sample = sample_positions(len(Xp))
        query = pd.concat([Xp.iloc[train_sample], Xtp.iloc[sample_positions(len(Xtp))]], ignore_index=True)
        key_columns = [clock, *general_entities]
        query = query[key_columns]
    if cfg["event_activity"]:
        source = pd.concat([X[key_columns], X_test[key_columns]], ignore_index=True)
        source[clock] = _time_values(source[clock])
        if cfg["audit_required"]:
            audit_event_kernel(lambda data, ask, _, __: event_activity(data, ask, clock, general_entities),
                               source, source[clock], query, clock, "event_activity", budget=budget,
                               train_rows=len(train_sample),
                               reference=lambda data, ask, _, __: activity_reference(data, ask, clock, general_entities))
        Xp = pd.concat([Xp, event_activity(source, Xp, clock, general_entities)], axis=1)
        Xtp = pd.concat([Xtp, event_activity(source, Xtp, clock, general_entities)], axis=1)

    def target_features(source, evaluation, observed, y):
        return causal_target_history(source, evaluation, clock, observed, y, general_entities,
                                     window=cfg["history_window"], smoothing=cfg["history_smoothing"],
                                     median=metric_obj.name == "mae")

    if cfg["causal_history"] and cfg["audit_required"]:
        first_fit, _, _ = _past_monitor(folds[0][0], labeled_times, y_fit, task_type, available_at)
        observed = (available_at if available_at is not None else labeled_times).iloc[first_fit]

        def reference(source, evaluation, known, labels, *, corrupt_future=False):
            return history_reference(source, evaluation, clock, known, labels, general_entities,
                                     window=cfg["history_window"], smoothing=cfg["history_smoothing"],
                                     median=metric_obj.name == "mae", corrupt_future=corrupt_future)

        audit_event_kernel(target_features, Xp.iloc[first_fit][key_columns], observed, query, clock,
                           "causal_history", labels=y_fit[first_fit], budget=budget, reference=reference,
                           train_rows=len(train_sample),
                           corrupted_reference=lambda *args: reference(*args, corrupt_future=True))
    if cfg["audit_required"]:
        audit_log("pass", "model_input", features=list(Xp.columns), rule="declared row values and audited transforms")
    category_probes = {c: [] for c in probe_columns}
    model_inputs = {}
    fold_diagnostics = []

    def fold_frames(tr_idx, va_idx, predict_idx=None):
        """Матрицы обучения, валидации и test фолда, с кодировками по таргету, если они заданы."""
        Xtr, Xva, Xte = Xp.iloc[tr_idx], Xp.iloc[va_idx], Xtp
        if temporal:
            parts = [Xva]
            if predict_idx is not None:
                parts.append(Xp.iloc[predict_idx])
            else:
                parts.append(Xte)
            evaluation = pd.concat(parts, ignore_index=True)
            history_cols = candidates["entity_history"] if cfg["entity_history"] else []
            if te_cols or history_cols:
                # Monitor, validation and test rows query the fit history but never enter it.
                eval_idx = va_idx if predict_idx is None else np.concatenate([va_idx, predict_idx])
                query_time = pd.concat([labeled_times.iloc[tr_idx], evaluation[clock]], ignore_index=True)
                observed = (available_at if available_at is not None else labeled_times).iloc[tr_idx]
                for col in [*te_cols, *history_cols]:
                    keys = te_keys[col] if col in te_keys else Xp[col].to_numpy()
                    test_keys = te_keys_test[col] if col in te_keys else Xtp[col].to_numpy()
                    parts = [keys[tr_idx], keys[eval_idx]]
                    if predict_idx is None:
                        parts.append(test_keys)
                    query_keys = np.concatenate(parts)
                    encoded = _past_encoding(keys[tr_idx], observed, te_Y[tr_idx], query_keys, query_time, smoothing=20)
                    names = _te_columns([col], te_Y.shape[1] + 1)
                    Xtr = _with_columns(Xtr, encoded[:len(tr_idx)], names)
                    evaluation = _with_columns(evaluation, encoded[len(tr_idx):], names)
            if cfg["entity_history"]:
                from features import add_fold_entity_history
                frequency_cols = [c for c in Xtr if _is_categorical(Xtr[c]) and c != clock]
                Xtr, evaluation = add_fold_entity_history(Xtr, evaluation, list(dict.fromkeys(history_cols + frequency_cols)), clock)
            if cfg["ad_history"]:
                from features import add_ad_history
                Xtr, evaluation = add_ad_history(Xtr, evaluation, y_fit[tr_idx], clock, ad_groups,
                                                  window=cfg["ad_history_window"], smoothing=cfg["ad_history_smoothing"])
            if cfg["causal_history"]:
                observed = (available_at if available_at is not None else labeled_times).iloc[tr_idx]
                target_frame = pd.concat([Xtr[[clock, *general_entities]], evaluation[[clock, *general_entities]]],
                                         ignore_index=True)
                derived = target_features(Xtr, target_frame, observed, y_fit[tr_idx])
                Xtr = pd.concat([Xtr, derived.iloc[:len(Xtr)].set_axis(Xtr.index)], axis=1)
                evaluation = pd.concat([evaluation, derived.iloc[len(Xtr):].set_axis(evaluation.index)], axis=1)
            if cfg["categorical_ids"]:
                Xtr = Xtr.copy()
                for col in candidates["categorical_ids"]:
                    Xtr[col] = Xtr[col].astype("category")
            Xtr, prepared, cats = _prepare_features(Xtr, evaluation, list(Xtr.columns), "native")
            if cfg["model"] == "catboost":
                if cfg["audit_required"]:
                    # Keep CatBoost from computing target statistics in a shuffled training order.
                    # High-cardinality categories use only the fold's unlabeled vocabulary.
                    ordinal = [col for col in cats if len(Xtr[col].cat.categories) >= _AUDITED_CAT_ONE_HOT]
                    for col in ordinal:
                        Xtr[col] = Xtr[col].cat.codes.astype("float32")
                        prepared[col] = prepared[col].cat.codes.astype("float32")
                    cats = [col for col in cats if col not in ordinal]
                extra["cat_cols"] = cats
                Xtr, prepared = _to_catboost_frame(Xtr, cats), _to_catboost_frame(prepared, cats)
            Xva, Xte = prepared.iloc[:len(va_idx)], prepared.iloc[len(va_idx):]
        if te_cols and not temporal:
            enc_tr, enc_va, enc_te = _target_encode(te_keys, te_keys_test, te_Y, tr_idx, va_idx,
                                                    groups, cfg["fold_seed"], te_periods)
            Xtr, Xva = _with_columns(Xtr, enc_tr, te_names), _with_columns(Xva, enc_va, te_names)
            Xte = _with_columns(Xte, enc_te, te_names)
        return Xtr, Xva, Xte

    # Переобучение на всем train идет после фолдов, под него держим долю бюджета
    refit_slot = 2.0 if temporal else (1.25 if cfg["refit_full"] else 0.0)

    logger.info("Старт: %s, %s, метрика %s, модель %s, %d фолдов, %d строк, %d признаков (%d кат.)%s",
                task_type, "log1p" if log_target else "raw", metric_obj.name, cfg["model"],
                len(folds), len(y_fit), Xp.shape[1], len(cat_cols),
                f", кодирование по таргету {te_cols}" if te_cols else "")

    n = len(y_fit)
    oof = np.full((n, n_classes), np.nan) if n_classes else np.full(n, np.nan)
    test_sum = np.zeros((len(Xtp), n_classes)) if n_classes else np.zeros(len(Xtp))
    fold_scores_raw, fold_scores, fold_times, best_iters, done_folds = [], [], [], [], []
    fold_baselines = []
    time_stopped_any = False
    monitor_windows = []

    for i, (tr_idx, va_idx) in enumerate(folds):
        now = time.monotonic()
        if now >= deadline:
            logger.warning("Бюджет времени исчерпан, выполнено %d из %d фолдов", i, len(folds))
            break
        # Равная доля оставшегося бюджета на фолд, 10%% в запасе на постобработку
        fold_deadline = now + 0.9 * (deadline - now) / (len(folds) - i + refit_slot)
        try:
            fit_idx, monitor_idx, fold_config = tr_idx, va_idx, cfg
            if temporal:
                fit_idx, monitor_idx, mode = _past_monitor(tr_idx, labeled_times, y_fit, task_type, available_at)
                monitor_windows.append({
                    "mode": mode, "fit_rows": len(fit_idx), "monitor_rows": len(monitor_idx),
                    "fit_max": str(labeled_times.iloc[fit_idx].max()),
                    "monitor_min": str(labeled_times.iloc[monitor_idx].min()),
                    "monitor_max": str(labeled_times.iloc[monitor_idx].max()),
                })
                if mode == "training_monitor":
                    rounds = min(64, int(cfg["num_boost_round"]))
                    fold_config = cfg | {"num_boost_round": rounds, "early_stopping_rounds": rounds + 1}
            Xtr, Xmon, Xpred = fold_frames(fit_idx, monitor_idx, va_idx if temporal else None)
            if not model_inputs:
                model_inputs = {c: {"input_dtype": str(Xp[c].dtype) if c in Xp else "derived",
                                    "model_dtype": str(Xtr[c].dtype),
                                    "cardinality_in_fit": int(Xtr[c].nunique())} for c in Xtr}
            fold_weight = training_weights(fit_idx)
            pva, pte, best_iter, stopped, last_iter = fit_fn(
                Xtr, y_train_scale[fit_idx], fold_weight, Xmon, y_train_scale[monitor_idx], Xpred,
                task_type, metric_obj, n_classes, fold_config, fold_deadline, **extra,
            )
            if temporal:
                pva = pte
        except Exception as e:
            raise RuntimeError(f"Ошибка обучения {cfg['model']} на фолде {i + 1}: "
                               f"{type(e).__name__}: {e}") from e
        if log_target:
            pva, pte = np.sign(pva) * np.expm1(np.abs(pva)), np.sign(pte) * np.expm1(np.abs(pte))
        if not np.isfinite(pva).all() or not np.isfinite(pte).all():
            raise ValueError("Model returned nonfinite predictions")
        prediction = pva[:, 1] if task_type == "binary" else pva
        fold_diagnostics.append({"fold": i, "best_iteration": int(best_iter), "last_iteration": int(last_iter),
                                 "stopped_for_time": bool(stopped), "prediction_std": float(prediction.std())})
        if probe_columns:
            prior = float(y_fit[fit_idx].mean())
            for col in probe_columns:
                entries = pd.DataFrame({"key": Xp[col].iloc[fit_idx].to_numpy(), "y": y_fit[fit_idx]})
                stats = entries.groupby("key", observed=True)["y"].agg(["sum", "count"])
                rates = (stats["sum"] + 20 * prior) / (stats["count"] + 20)
                held = Xp[col].iloc[va_idx]
                values = held.map(rates).astype(float)
                score = average_precision_score(y_fit[va_idx], values.fillna(prior))
                # Missing keys form a separate bucket in the total-variation distance.
                fit_missing, validation_missing = entries["key"].isna().mean(), held.isna().mean()
                fit_share = stats["count"] / len(fit_idx)
                held_share = held.value_counts(dropna=True) / len(va_idx)
                tv = (fit_share.sub(held_share, fill_value=0.).abs().sum() + abs(fit_missing - validation_missing)) / 2
                category_probes[col].append({
                    "fold": i, "ap": float(score), "baseline": float(y_fit[va_idx].mean()),
                    "known_rows": int(values.notna().sum()), "validation_rows": len(va_idx),
                    "unseen_nonmissing_share": float((held.notna() & values.isna()).mean()),
                    "fit_missing_share": float(fit_missing), "validation_missing_share": float(validation_missing),
                    "frequency_total_variation": float(tv),
                })
        fold_scores_raw.append(_score_fold(y_fit[va_idx], pva, metric_obj))
        if clip:
            # Диапазон только по train фолда, чтобы скор оставался честным
            fold_scores.append(_score_fold(
                y_fit[va_idx], np.clip(pva, y_fit[tr_idx].min(), y_fit[tr_idx].max()), metric_obj))
        else:
            fold_scores.append(fold_scores_raw[-1])
        if metric_obj.name in ("pr_auc", "mae"):
            constant = float(y_fit[tr_idx].mean() if metric_obj.name == "pr_auc" else np.median(y_fit[tr_idx]))
            guess = np.full(len(va_idx), constant)
            baseline = (float(average_precision_score(y_fit[va_idx], guess)) if metric_obj.name == "pr_auc"
                        else float(np.abs(y_fit[va_idx] - guess).mean()))
            fold_baselines.append({"fold": i + 1, "constant": constant, "score": baseline,
                                   "train_rows": len(tr_idx), "validation_rows": len(va_idx)})
        oof[va_idx] = pva
        if not temporal:
            test_sum += pte
        time_stopped_any |= stopped
        done_folds.append(i)
        best_iters.append(int(best_iter))
        fold_times.append(time.monotonic() - now)

        logger.info("Фолд %d/%d: %s=%.5f, итераций %d, %.1f с%s", i + 1, len(folds), metric_obj.name,
                    fold_scores[-1], best_iter, fold_times[-1],
                    " (урезано по времени)" if stopped else "")

    if time_stopped_any:
        logger.warning("Часть фолдов остановлена по бюджету времени, раунды урезаны")
    if not done_folds:
        raise TimeoutError("Training budget expired before the first fold")
    test_pred = test_sum / len(done_folds)

    # Модель на всем train: данных на 1/n_folds больше, чем у моделей фолдов. Скор CV она
    # не меняет, поэтому в test идет пополам со средним фолдов, а не вместо него
    refit_info = None
    if temporal or (cfg["refit_full"] and len(done_folds) == len(folds) and not time_stopped_any):
        left = deadline - time.monotonic()
        if temporal or left > refit_slot * float(np.mean(fold_times)):
            t_refit = time.monotonic()
            rounds = max(1, round(np.mean(best_iters) * 1.1)) if temporal else max(_MIN_ROUNDS, int(np.mean(best_iters) * _REFIT_ROUNDS))
            # Ранней остановки нет: число раундов известно из фолдов. Валидация нужна бэкендам
            # формально, берем небольшой кусок train
            rcfg = cfg | {"num_boost_round": rounds, "early_stopping_rounds": rounds + 1, "final_fit": True}
            all_idx = refit_idx
            monitor = np.random.default_rng(cfg["seed"]).choice(all_idx, min(len(all_idx), 2000), replace=False)
            try:
                Xall, Xmon, Xte = fold_frames(all_idx, monitor)
                _, pte, _, stopped, _ = fit_fn(
                    Xall, y_train_scale[all_idx], training_weights(all_idx), Xmon, y_train_scale[monitor], Xte,
                    task_type, metric_obj, n_classes, rcfg, deadline, **extra,
                )
                if log_target:
                    pte = np.sign(pte) * np.expm1(np.abs(pte))
                if not np.isfinite(pte).all():
                    raise ValueError("Refit returned nonfinite predictions")
                if temporal:
                    test_pred = pte
                elif not stopped:
                    test_pred = 0.5 * test_pred + 0.5 * pte
                refit_info = {"rounds": rounds, "applied": temporal or not stopped,
                              "time": time.monotonic() - t_refit}
                logger.info("Модель на всем train: %d раундов, %.1f с%s", rounds, refit_info["time"],
                            "" if not stopped else ", остановлена по времени, не используется")
            except Exception as e:  # noqa: BLE001 - без нее остается среднее фолдов
                logger.exception("Full-history refit failed")
                if temporal:
                    raise
        else:
            logger.info("Модель на всем train пропущена: осталось %.0f с", left)
    t_post = time.monotonic()
    score_raw = float(np.nanmean(fold_scores_raw))
    postprocess_info = None
    weights = np.ones(n_classes) if n_classes else None

    # Клиппинг регрессии оставляем, только если он улучшает скор по фолдам
    if clip:
        score_clip = float(np.nanmean(fold_scores))
        # Temporal windows apply the configured rule without selecting it on future outcomes.
        clip = temporal or _is_better(score_clip, score_raw, metric_obj.greater_is_better)
        logger.info("Клиппинг в диапазон таргета: %s %.5f -> %.5f, %s", metric_obj.name, score_raw,
                    score_clip, "применяю" if clip else "не улучшает, отключаю")
        if clip:
            for i in done_folds:
                tr_idx, va_idx = folds[i]
                oof[va_idx] = np.clip(oof[va_idx], y_fit[tr_idx].min(), y_fit[tr_idx].max())
        else:
            fold_scores = list(fold_scores_raw)
        postprocess_info = {"type": "clip", "applied": clip, "score_after": score_clip,
                            "range": [float(y_fit.min()), float(y_fit.max())]}

    # Постобработка для метрик по меткам: веса классов, честная оценка через кросс-фиттинг
    if (not temporal and task_type in CLASSIFICATION and not metric_obj.needs_proba and cfg["postprocess"]
            and len(done_folds) >= 2):
        weights, cross_scores, postprocess_info = _cross_fit_class_weights(
            oof, y_fit, [folds[i] for i in done_folds], metric_obj, score_raw)
        postprocess_info["weights"] = dict(zip(
            encoder.classes_.tolist(), postprocess_info.pop("weights_raw").round(4).tolist()))
        if cross_scores is not None:
            fold_scores = cross_scores

    # Финальные предсказания
    if task_type == "regression":
        predictions = _finalize_regression(test_pred, y_fit[refit_idx], clip)
        if temporal and metric_obj.name == "mae" and cfg["mae_shift"] and len(done_folds) == len(folds):
            adjusted, shifted_scores, shifts = causal_mae_shift(oof, y_fit, folds, labeled_times, available_at, test_cutoff)
            if sum(a < b for a, b in zip(shifted_scores, fold_scores)) > len(fold_scores) / 2 and np.mean(shifted_scores) < np.mean(fold_scores):
                oof, fold_scores = adjusted, shifted_scores
                predictions = _finalize_regression(predictions + shifts[-1], y_fit[refit_idx], clip)
                postprocess_info = {"type": "causal_mae_shift", "shifts": shifts}
        probabilities, class_order = None, None
    else:
        probabilities = test_pred
        predictions = encoder.inverse_transform(_apply_weights(test_pred, weights))
        class_order = encoder.classes_.tolist()

    # OOF в разрезе исходных строк X
    if labeled.all():
        oof_full = oof
    else:
        oof_full = np.full((len(X),) + oof.shape[1:], np.nan)
        oof_full[labeled] = oof

    meta_known_from = None
    if temporal:
        windows = []
        for i in done_folds:
            train_idx, valid_idx = folds[i]
            latest = labeled_times.iloc[train_idx].max()
            earliest = labeled_times.iloc[valid_idx].min()
            if not latest < earliest:
                raise ValueError("Training time overlaps its validation window")
            windows.append({"train_rows": len(train_idx), "validation_rows": len(valid_idx),
                            "train_max": str(latest), "validation_min": str(earliest)})
            if available_at is not None:
                details = cutoff_details[i]
                cutoff = details["cutoff"]
                windows[-1].update(validation_cutoff=cutoff.isoformat() if isinstance(cutoff, pd.Timestamp) else str(cutoff),
                                   train_pending=details["train_pending"],
                                   validation_pending_at_prediction=details["validation_pending_at_prediction"])
        if available_at is not None:
            cfg["prediction_cutoff"] = test_cutoff.isoformat() if isinstance(test_cutoff, pd.Timestamp) else str(test_cutoff)
            cfg["observed_before_prediction"] = len(refit_idx)
            cutoffs = pd.Index([d["cutoff"] for d in cutoff_details])
            meta_known_from = np.full(len(labeled), len(cutoffs), dtype=np.int32)
            meta_known_from[labeled] = cutoffs.searchsorted(available_at, side="right")
        cfg["fold_windows"] = windows
        cfg["early_stopping_windows"] = monitor_windows
    fit_time = time.monotonic() - t_start
    score = float(np.nanmean(fold_scores))
    score_std = float(np.nanstd(fold_scores))
    cfg.update(task_type=task_type, metric=metric_obj.name, n_folds_done=len(done_folds),
               time_budget=time_budget, fold_baselines=fold_baselines)
    logger.info("Итог: %s=%.5f +- %.5f (до постобработки %.5f), %.1f с",
                metric_obj.name, score, score_std, score_raw, fit_time)

    return {
        "score": score,
        "score_std": score_std,
        "score_raw": score_raw,
        "predictions": predictions,
        "probabilities": probabilities,
        "class_order": class_order,
        "oof": oof_full,
        "fit_time": fit_time,
        "config": cfg,
        # Дополнительно для журнала экспериментов
        "postprocess_time": time.monotonic() - t_post,
        "fold_ids": _fold_ids(folds, done_folds, labeled),
        "meta_known_from": meta_known_from,
        "fold_scores": [float(s) for s in fold_scores],
        "fold_scores_raw": [float(s) for s in fold_scores_raw],
        "fold_times": fold_times,
        "best_iterations": best_iters,
        "feature_candidates": candidates | {"ad_history": bool(ad_groups), "event_activity": datetime_clock,
                                            "causal_history": bool(general_entities) and task_type != "multiclass"},
        "model_inputs": model_inputs,
        "fold_diagnostics": fold_diagnostics,
        "category_probes": ({"smoothing_rows": 20, "source": "labels in model-fit prefix before monitor",
                             "availability_assumption": "fit-prefix labels are observed at the prediction cutoff",
                             "columns": category_probes} if probe_columns else None),
        "greater_is_better": metric_obj.greater_is_better,
        "postprocess": postprocess_info,
        "refit": refit_info,
    }


def causal_mae_shift(oof, y, folds, times, available, test_cutoff):
    """Fit each shift to residuals from previously observable validation labels."""
    ready = available if available is not None else times
    adjusted = oof.copy()
    scores, shifts, seen = [], [], []
    for _, val in folds:
        eligible = np.asarray(seen, dtype=int)
        eligible = eligible[ready.iloc[eligible].lt(times.iloc[val].min()).to_numpy()]
        shift = float(np.median(y[eligible] - oof[eligible])) if len(eligible) else 0.
        adjusted[val] += shift
        shifts.append(shift)
        scores.append(float(np.abs(y[val] - adjusted[val]).mean()))
        seen.extend(val.tolist())
    eligible = np.asarray(seen, dtype=int)
    if test_cutoff is not None:
        eligible = eligible[ready.iloc[eligible].lt(test_cutoff).to_numpy()]
    shifts.append(float(np.median(y[eligible] - oof[eligible])) if len(eligible) else 0.)
    return adjusted, scores, shifts


def _fold_ids(folds: list, done_folds: list, labeled: np.ndarray) -> np.ndarray:
    """Номер фолда валидации для каждой строки X, -1 для строк без таргета и невыполненных фолдов."""
    ids_lab = np.full(int(labeled.sum()), -1, dtype=np.int16)
    for i in done_folds:
        ids_lab[folds[i][1]] = i
    ids = np.full(len(labeled), -1, dtype=np.int16)
    ids[labeled] = ids_lab
    return ids


def _finalize_regression(pred: np.ndarray, y_fit: np.ndarray, clip: bool) -> np.ndarray:
    pred = pred.copy()
    bad = ~np.isfinite(pred)
    if bad.any():
        logger.warning("В предсказаниях %d нечисловых значений, заменяю средним таргета", bad.sum())
        pred[bad] = float(np.mean(y_fit))
    if clip:
        pred = np.clip(pred, y_fit.min(), y_fit.max())
    return pred


def _score_fold(y_true: np.ndarray, pred: np.ndarray, metric: Metric) -> float:
    if pred.ndim == 2 and not metric.needs_proba:
        pred = pred.argmax(1)
    try:
        return metric.func(y_true, pred)
    except ValueError as e:
        # Например, roc_auc на фолде без одного из классов
        logger.warning("Метрика %s не посчиталась на фолде: %s", metric.name, e)
        return float("nan")


# ---------------------------------------------------------------------------
# Бленд результатов
# ---------------------------------------------------------------------------

@dataclass
class _BlendData:
    task_type: str
    metric: Metric
    labeled: np.ndarray           # маска строк y с таргетом
    y_enc: np.ndarray             # таргет по размеченным строкам, классы как индексы class_order
    class_order: list | None
    oofs: np.ndarray              # (моделей, строк[, классов])
    tests: np.ndarray             # (моделей, строк test[, классов])
    folds: list                   # только фолды с полными OOF у всех моделей
    cfg0: dict
    meta_known_from: np.ndarray | None


def _prepare_blend(results: list[dict], y: pd.Series, metric: str) -> _BlendData:
    if not results:
        raise ValueError("Пустой список результатов для бленда")
    task_types = {r["config"]["task_type"] for r in results}
    if len(task_types) != 1:
        raise ValueError(f"Нельзя блендить разные типы задач: {task_types}")
    task_type = task_types.pop()
    metric_obj = get_metric(metric, task_type)

    y_series = pd.Series(np.asarray(y))
    labeled = y_series.notna().to_numpy()
    y_lab = y_series[labeled]
    class_order = results[0]["class_order"]
    if task_type == "regression":
        y_enc = pd.to_numeric(y_lab).to_numpy(dtype=float)
        test_key = "predictions"
    else:
        if any(r["class_order"] != class_order for r in results):
            raise ValueError("У результатов разный порядок классов, бленд невозможен")
        y_enc = y_lab.map({c: i for i, c in enumerate(class_order)})
        if y_enc.isna().any():
            raise ValueError("В y есть классы, которых нет в class_order")
        y_enc = y_enc.to_numpy(dtype=int)
        test_key = "probabilities"

    oofs = np.stack([np.asarray(r["oof"], dtype=float)[labeled] for r in results])
    tests = np.stack([np.asarray(r[test_key], dtype=float) for r in results])
    if oofs.shape[1] != len(y_enc):
        raise ValueError("Длина OOF не совпадает с длиной y")

    cfg0 = results[0]["config"]
    if any((r["config"].get("split") or {}) != (cfg0.get("split") or {}) for r in results):
        raise ValueError("Cannot blend results with different split strategies")
    meta_known_from = results[0].get("meta_known_from")
    if any(not np.array_equal(r.get("meta_known_from"), meta_known_from) for r in results):
        raise ValueError("Cannot blend results with different label availability")
    if "fold_ids" in results[0]:
        fold_ids = np.asarray(results[0]["fold_ids"])
        if any("fold_ids" not in r or not np.array_equal(r["fold_ids"], fold_ids) for r in results):
            raise ValueError("Cannot blend results with different validation rows")
        fid = fold_ids[labeled]
        temporal = (cfg0.get("split") or {}).get("type") == "time"
        folds = []
        for k in np.unique(fid[fid >= 0]):
            fit = (fid >= 0) & ((fid < k) if temporal else (fid != k))
            if meta_known_from is not None:
                fit &= meta_known_from[labeled] <= k
            folds.append((np.flatnonzero(fit), np.flatnonzero(fid == k)))
    else:
        if cfg0.get("split") or any("fold_ids" in r for r in results):
            raise ValueError("Missing validation fold IDs")
        folds = _make_folds(y_enc, task_type, cfg0["n_folds"], cfg0.get("fold_seed", cfg0["seed"]))
    finite = np.isfinite(oofs).reshape(len(results), len(y_enc), -1).all(axis=(0, 2))
    if not folds or any(not finite[va].all() for _, va in folds):
        raise ValueError("Missing predictions on validation rows")
    return _BlendData(task_type, metric_obj, labeled, y_enc, class_order, oofs, tests, folds, cfg0, meta_known_from)


def _finish_blend(data: _BlendData, oof: np.ndarray, test_pred: np.ndarray, config: dict,
                  t_start: float, members_fit_time: float) -> dict:
    """Скор по фолдам, постобработка и итоговый словарь в формате train_and_predict."""
    metric_obj, y_enc, folds = data.metric, data.y_enc, data.folds
    oof = oof.copy()
    fold_scores_raw = [_score_fold(y_enc[va], oof[va], metric_obj) for _, va in folds]
    score_raw = float(np.nanmean(fold_scores_raw))
    fold_scores = fold_scores_raw
    postprocess_info = None

    temporal = (data.cfg0.get("split") or {}).get("type") == "time"
    if data.task_type == "regression" and temporal:
        predictions = _finalize_regression(test_pred, y_enc, False)
        probabilities = None
    elif data.task_type == "regression":
        # Клиппинг по диапазону train фолда, оставляем только при улучшении
        clipped = [np.clip(oof[va], y_enc[tr].min(), y_enc[tr].max()) for tr, va in folds]
        clip_scores = [_score_fold(y_enc[va], p, metric_obj) for (_, va), p in zip(folds, clipped)]
        clip = _is_better(float(np.nanmean(clip_scores)), score_raw, metric_obj.greater_is_better)
        if clip:
            fold_scores = clip_scores
            for (_, va), p in zip(folds, clipped):
                oof[va] = p
        postprocess_info = {"type": "clip", "applied": clip,
                            "score_after": float(np.nanmean(clip_scores))}
        predictions = _finalize_regression(test_pred, y_enc, clip)
        probabilities = None
    else:
        cls_w = np.ones(len(data.class_order))
        if not temporal and not metric_obj.needs_proba and len(folds) >= 2:
            cls_w, cross_scores, postprocess_info = _cross_fit_class_weights(
                oof, y_enc, folds, metric_obj, score_raw)
            postprocess_info["weights"] = dict(zip(
                data.class_order, postprocess_info.pop("weights_raw").round(4).tolist()))
            if cross_scores is not None:
                fold_scores = cross_scores
        probabilities = test_pred
        predictions = np.asarray(data.class_order, dtype=object)[_apply_weights(test_pred, cls_w)]

    if data.labeled.all():
        oof_full = oof
    else:
        oof_full = np.full((len(data.labeled),) + oof.shape[1:], np.nan)
        oof_full[data.labeled] = oof

    cfg0 = data.cfg0
    config = config | {"task_type": data.task_type, "metric": metric_obj.name, "split": cfg0.get("split"),
                       "n_folds": cfg0["n_folds"], "n_folds_done": len(folds), "seed": cfg0["seed"],
                       "fold_seed": cfg0.get("fold_seed", cfg0["seed"])}
    if temporal and cfg0.get("fold_windows"):
        config["fold_windows"] = cfg0["fold_windows"]
        for key in ("early_stopping_windows", "label_delay_unit", "prediction_cutoff", "observed_before_prediction"):
            config[key] = cfg0.get(key)
    score, score_std = float(np.nanmean(fold_scores)), float(np.nanstd(fold_scores))
    config["fold_baselines"] = data.cfg0.get("fold_baselines", [])
    return {
        "score": score,
        "score_std": score_std,
        "score_raw": score_raw,
        "predictions": predictions,
        "probabilities": probabilities,
        "class_order": data.class_order,
        "oof": oof_full,
        "fit_time": time.monotonic() - t_start,
        "config": config,
        "fold_scores": [float(s) for s in fold_scores],
        "fold_scores_raw": [float(s) for s in fold_scores_raw],
        "greater_is_better": metric_obj.greater_is_better,
        "postprocess": postprocess_info,
        "members_fit_time": members_fit_time,
        "fold_ids": _fold_ids(folds, list(range(len(folds))), data.labeled),
        "meta_known_from": data.meta_known_from,
    }


def blend_results(results: list[dict], y: pd.Series, metric: str,
                  weights: list[float] | None = None) -> dict:
    """Бленд с заданными весами (по умолчанию равными) OOF и test нескольких результатов.

    Для классификации усредняются вероятности и заново подбираются веса классов
    через кросс-фиттинг. Скор считается по фолдам первого результата.
    """
    t_start = time.monotonic()
    data = _prepare_blend(results, y, metric)
    w = np.ones(len(results)) if weights is None else np.asarray(weights, dtype=float)
    if len(w) != len(results) or (w < 0).any() or w.sum() <= 0:
        raise ValueError("Веса бленда должны быть неотрицательны и совпадать по длине с results")
    w = w / w.sum()
    res = _finish_blend(data, np.tensordot(w, data.oofs, axes=1), np.tensordot(w, data.tests, axes=1),
                        {"blend_of": [r["config"] for r in results], "blend_weights": w.tolist()},
                        t_start, float(sum(r["fit_time"] for r in results)))
    logger.info("Бленд %d результатов: %s=%.5f +- %.5f (до постобработки %.5f)", len(results),
                res["config"]["metric"], res["score"], res["score_std"], res["score_raw"])
    return res


def _blend_objective(data: _BlendData) -> Metric:
    # Метрика по меткам кусочно-постоянна, для весов бленда берем гладкий logloss
    if data.task_type in CLASSIFICATION and not data.metric.needs_proba:
        return METRICS["logloss"]
    return data.metric


def _greedy_blend_weights(oofs: np.ndarray, y: np.ndarray, objective: Metric,
                          n_iter: int = 50, patience: int = 10) -> np.ndarray:
    """Жадный отбор моделей с повторениями (Caruana): веса неотрицательны, сумма 1."""
    counts = np.zeros(len(oofs))
    total = np.zeros_like(oofs[0])
    best_score, best_counts, stale = None, None, 0
    for it in range(1, n_iter + 1):
        step_score, step_model = None, 0
        for m in range(len(oofs)):
            s = _score_fold(y, (total + oofs[m]) / it, objective)
            if step_score is None or _is_better(s, step_score, objective.greater_is_better):
                step_score, step_model = s, m
        total += oofs[step_model]
        counts[step_model] += 1
        if best_score is None or _is_better(step_score, best_score, objective.greater_is_better):
            best_score, best_counts, stale = step_score, counts.copy(), 0
        else:
            stale += 1
            if stale >= patience:
                break
    return best_counts / best_counts.sum()


def optimize_blend_weights(results: list[dict], y: pd.Series, metric: str) -> dict:
    """Взвешенный бленд с честной оценкой через кросс-фиттинг.

    Для каждого фолда веса подбираются на OOF остальных фолдов и применяются к отложенному.
    Веса для test подбираются на всех OOF. Если бленд не лучше лучшей одиночной модели,
    возвращается она сама.
    """
    t_start = time.monotonic()
    data = _prepare_blend(results, y, metric)
    objective = _blend_objective(data)
    greater = data.metric.greater_is_better
    best_single = max(results, key=lambda r: r["score"] if greater else -r["score"])
    if len(results) == 1:
        return dict(best_single)

    done_mask = np.zeros(len(data.y_enc), dtype=bool)
    for _, va in data.folds:
        done_mask[va] = True
    oof_cf = np.full_like(data.oofs[0], np.nan)
    fold_weights = []
    for fit_idx, va in data.folds:
        if len(fit_idx):
            w = _greedy_blend_weights(data.oofs[:, fit_idx], data.y_enc[fit_idx], objective)
        else:
            w = np.full(len(results), 1 / len(results))
        oof_cf[va] = np.tensordot(w, data.oofs[:, va], axes=1)
        fold_weights.append(w.round(4).tolist())
    done_idx = np.flatnonzero(done_mask)
    w_all = _greedy_blend_weights(data.oofs[:, done_idx], data.y_enc[done_idx], objective)

    members_time = float(sum(r["fit_time"] for r in results))
    configs = [r["config"] for r in results]
    weighted = _finish_blend(
        data, oof_cf, np.tensordot(w_all, data.tests, axes=1),
        {"blend_of": configs, "blend_weights": w_all.round(4).tolist(), "blend_type": "weighted",
         "blend_fold_weights": fold_weights, "blend_objective": objective.name},
        t_start, members_time)
    # Равные веса ничего не подбирают, их оценка честная и часто лучше шумных подобранных весов
    w_eq = np.full(len(results), 1 / len(results))
    equal = _finish_blend(
        data, np.tensordot(w_eq, data.oofs, axes=1), np.tensordot(w_eq, data.tests, axes=1),
        {"blend_of": configs, "blend_weights": w_eq.round(4).tolist(), "blend_type": "equal"},
        t_start, members_time)
    logger.info("Бленд %d моделей: взвешенный %s=%.5f (веса %s), равный %.5f, лучшая одиночная %.5f",
                len(results), data.metric.name, weighted["score"], w_all.round(3).tolist(),
                equal["score"], best_single["score"])
    res = weighted if _is_better(weighted["score"], equal["score"], greater) else equal
    if not compare_results(best_single, res)["accept"] or not _is_better(res["score"], best_single["score"], greater):
        logger.info("Бленд не лучше лучшей одиночной модели, возвращаю ее")
        return best_single | {"blend_rejected_score": res["score"]}
    return res


def compare_results(a: dict, b: dict) -> dict:
    """Попарное сравнение b с a по одинаковым фолдам.

    diffs положительны, когда b лучше. accept=True, только если b лучше на большинстве фолдов
    и средняя разница больше половины std разниц.
    """
    if ("fold_ids" in a or "fold_ids" in b) and not np.array_equal(a.get("fold_ids"), b.get("fold_ids")):
        raise ValueError("Cannot compare results from different validation rows")
    if a.get("fold_signature") != b.get("fold_signature"):
        raise ValueError("Cannot compare results from different validation rows")
    fa, fb = np.asarray(a["fold_scores"], float), np.asarray(b["fold_scores"], float)
    if len(fa) != len(fb):
        raise ValueError(f"Разное число фолдов: {len(fa)} и {len(fb)}, сравнение некорректно")
    ca, cb = a.get("config", {}), b.get("config", {})
    if (a.get("fold_signature") is None and "fold_ids" not in a
            and ca.get("fold_seed", ca.get("seed")) != cb.get("fold_seed", cb.get("seed"))):
        raise ValueError("Cannot compare results from different validation seeds")
    greater = a.get("greater_is_better", True)
    diffs = fb - fa if greater else fa - fb
    n_better = int((diffs > 0).sum())
    diff_mean = float(diffs.mean())
    diff_std = float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0
    accept = n_better > len(diffs) / 2 and diff_mean > 0.5 * diff_std and diff_mean > 0
    return {"accept": bool(accept), "diff_mean": diff_mean, "diff_std": diff_std,
            "n_better": n_better, "n_folds": len(diffs), "diffs": diffs.round(6).tolist()}


_AUTO_MIN_STEP = 20.0             # меньше стольких секунд на модель не запускаем (не больше 10% бюджета)
_AUTO_MAX_SEEDS = 3               # максимум моделей одного типа с разными сидами
_AUTO_SAFETY = 1.3                # запас к оценке времени следующей модели
_AUTO_UNSEEN_MAX = 0.01           # при большей доле новых категорий в test XGBoost не используем
_AUTO_TUNE_MIN_TRIALS = 8         # подбор параметров, только если влезает столько обучений LightGBM
_AUTO_TUNE_MAX_TRIALS = 40        # и не дольше стольких обучений


def tune_backend(model, X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame, task_type: str, metric: str,
                 config: dict, time_budget: float) -> dict:
    """Search before the first outer validation window and rescore winners on outer OOF."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    deadline = time.monotonic() + time_budget
    greater = get_metric(metric, task_type).greater_is_better
    base = dict(config) | {"model": model, "postprocess": False, "refit_full": False}


    split = base.get("split")
    if "split" not in base:
        from data_tools import detect_time_column
        clock = detect_time_column(X, X_test)
        split = {"type": "time", "column": clock} if clock else None
    temporal = (split or {}).get("type") == "time"
    if temporal:
        clock = _time_values(X[split["column"]]).reset_index(drop=True)
        targets = pd.Series(np.asarray(y))
        eligible = targets.notna() & clock.notna()
        times = pd.Index(clock[eligible].unique()).sort_values()
        if len(times) < 2:
            return {}
        n_blocks = min(len(times), max(2, int(base.get("n_folds", DEFAULT_CONFIG["n_folds"])) + 1))
        first_validation = times[(len(times) + n_blocks - 1) // n_blocks]
        prefix = eligible & clock.lt(first_validation)
        positions = np.flatnonzero(prefix)
        if base.get("label_delay_unit"):
            published = _label_available_at(clock.iloc[positions], targets.iloc[positions].to_numpy(),
                                            base["label_delay_unit"])
            positions = positions[published.lt(first_validation).to_numpy()]
        X = X.iloc[positions].reset_index(drop=True)
        y = targets.iloc[positions].reset_index(drop=True)
        if len(y) < 12 or X[split["column"]].nunique() < 4 or (task_type in CLASSIFICATION and y.nunique() < 2):
            return {}
        # Use a historical placeholder with the real as-of cutoff. Physical row order
        # within the training table must not set the outcome publication deadline.
        X_test = X.iloc[[-1]].copy()
        X_test[split["column"]] = first_validation
        base.update(n_folds=3, split=split)

    def suggest(trial):
        if model == "catboost":
            return {"learning_rate": trial.suggest_float("learning_rate", .025, .13, log=True),
                    "depth": trial.suggest_int("depth", 4, 8),
                    "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1., 24., log=True)}
        if model == "xgboost":
            return {"eta": trial.suggest_float("eta", .025, .13, log=True),
                    "max_depth": trial.suggest_int("max_depth", 3, 8),
                    "min_child_weight": trial.suggest_int("min_child_weight", 2, 32, log=True),
                    "lambda": trial.suggest_float("lambda", .25, 24., log=True)}
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 200, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 30.0, log=True),
        }

    fixed_params = {"bagging_freq": 1} if model == "lightgbm" else {}

    def objective(trial):
        left = deadline - time.monotonic()
        estimate = trial.study.user_attrs.get("trial_seconds", 10.)
        if left < 1.3 * estimate:
            trial.study.stop()
            raise optuna.TrialPruned()
        started = time.monotonic()
        params = suggest(trial)
        res = train_and_predict(X, y, X_test, task_type, metric, base | {"params": params | fixed_params},
                                time_budget=min(left, max(20., time_budget / 3)) if model != "lightgbm" else left)
        trial.study.set_user_attr("trial_seconds", max(1., time.monotonic() - started))
        # Фолды, урезанные по времени, дают несравнимый скор
        if res["config"]["n_folds_done"] < res["config"]["n_folds"]:
            raise optuna.TrialPruned()
        return res["score"]

    study = optuna.create_study(direction="maximize" if greater else "minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    if model == "lightgbm":
        study.enqueue_trial({"learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 20,
                             "feature_fraction": 0.8, "bagging_fraction": 0.8, "lambda_l2": 1.0})
    t0 = time.monotonic()
    study.optimize(objective, timeout=max(0.0, deadline - time.monotonic()),
                   n_trials=12 if temporal else None, catch=(ValueError, RuntimeError, TimeoutError))
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not done:
        logger.warning("Подбор параметров: ни одна попытка не завершилась")
        return {}
    logger.info("Подбор параметров %s: %d попыток за %.0f с, лучший %s=%.5f, параметры %s",
                model, len(done), time.monotonic() - t0, metric, study.best_value,
                {k: round(v, 4) if isinstance(v, float) else v for k, v in study.best_params.items()})
    return dict(study.best_params) | fixed_params


def tune_lightgbm(X, y, X_test, task_type, metric, config, time_budget):
    return tune_backend("lightgbm", X, y, X_test, task_type, metric, config, time_budget)


def auto_train(X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame, task_type: str, metric: str,
               time_budget: float, config: dict | None = None, use_catboost: bool = False,
               max_seeds: int = _AUTO_MAX_SEEDS) -> dict:
    """Рецепт: LightGBM и XGBoost, затем их сиды, пока укладываемся, затем взвешенный бленд.

    Все модели на одних и тех же фолдах. Перед каждой моделью оцениваем ее время по прошлому
    запуску того же типа и держим запас под финальный бленд, поэтому на больших данных
    рецепт обучает меньше моделей, а не выходит за бюджет. CatBoost по умолчанию выключен:
    на публичных задачах он долгий и почти ничего не добавляет. С use_catboost=True он идет
    в конце, если осталось больше половины бюджета. config - общие ключи для всех моделей.
    Возвращает словарь в формате train_and_predict плюс members и member_results.
    """
    t_start = time.monotonic()
    deadline = t_start + time_budget
    base = dict(config or {})
    seed0 = base.get("seed", DEFAULT_CONFIG["seed"])
    base.setdefault("fold_seed", seed0)
    remaining = lambda: deadline - time.monotonic()
    members: dict[str, dict] = {}
    kind_time: dict[str, float] = {}
    blend_time = [0.0]
    skipped_budget = []
    incomplete = []
    min_step = min(_AUTO_MIN_STEP, 0.1 * time_budget)

    def blend_estimate() -> float:
        # До первого замера оцениваем бленд по постобработке моделей: в нем она идет дважды
        post = max((r.get("postprocess_time", 0.0) for r in members.values()), default=0.0)
        return max(blend_time[0], 3.0 * post)

    def reserve() -> float:
        # Запас под финальный бленд, растет вместе с оценкой времени бленда
        return max(0.05 * time_budget, 1.5 * blend_estimate())

    def run(name: str, extra: dict, budget: float | None = None, force: bool = False) -> bool:
        avail = remaining() - reserve()
        if force:
            # Первая модель обязательна: без нее нет результата
            avail = max(avail, min(remaining(), min_step))
        budget = avail if budget is None else min(budget, avail)
        # Медленные модели учатся в разы дольше, их время оцениваем отдельно
        kind = extra.get("model", "lightgbm") + ("_slow" if name.endswith("_slow") else "")
        est = kind_time.get(kind)
        if not force:
            if budget < min_step:
                skipped_budget.append(name)
                logger.info("auto_train: %s пропущен, доступно %.0f с", name, budget)
                return False
            if est is not None and est * _AUTO_SAFETY > budget:
                skipped_budget.append(name)
                logger.info("auto_train: %s пропущен, оценка %.0f с больше доступных %.0f с",
                            name, est * _AUTO_SAFETY, budget)
                return False
        backend_params = base.get("params", {}) if extra.get("model") == base.get("model", "lightgbm") else {}
        cfg = base | extra | {"params": backend_params | extra.get("params", {})}
        logger.info("auto_train: %s, бюджет %.0f с, осталось %.0f с", name, budget, remaining())
        try:
            res = train_and_predict(X, y, X_test, task_type, metric, cfg, time_budget=budget)
        except Exception:
            incomplete.append(name)
            logger.exception("auto_train: модель %s упала, продолжаю без нее", name)
            return False
        if res["config"]["n_folds_done"] != res["config"]["n_folds"]:
            incomplete.append(name)
            logger.info("auto_train: %s missed validation folds", name)
            return False
        members[name] = res
        kind_time[kind] = res["fit_time"]
        return True

    def blend(previous: dict | None) -> dict | None:
        if len(members) < 2:
            return previous
        if remaining() < 1.2 * blend_estimate():
            logger.warning("auto_train: на бленд не хватает времени, оставляю прошлый результат")
            return previous
        t0 = time.monotonic()
        try:
            res = optimize_blend_weights(list(members.values()), y, metric)
        except Exception:
            logger.exception("auto_train: бленд упал, оставляю прошлый результат")
            return previous
        blend_time[0] = time.monotonic() - t0
        res["config"] = res["config"] | {"members": list(members)}
        if previous is not None and not _is_better(res["score"], previous["score"],
                                                   res["greater_is_better"]):
            return previous
        return res

    # XGBoost катастрофически ошибается на категориях, которых не было в train, а CV этого
    # не видит: в OOF новых категорий не бывает. Поэтому при их заметной доле обходимся без него
    features = base.get("features") or list(X.columns)
    unseen = unseen_category_share(X[features], X_test[features])
    use_lgb = backend_available("lightgbm")
    use_xgb = backend_available("xgboost") and max(unseen.values(), default=0.0) < _AUTO_UNSEEN_MAX
    use_cat = backend_available("catboost")
    if not use_xgb:
        logger.warning("auto_train: в test новые категории %s, XGBoost не используем",
                       {k: round(v, 3) for k, v in unseen.items() if v >= _AUTO_UNSEEN_MAX})

    # Шаг 1: два бустинга с дефолтами, LightGBM первым как самый быстрый
    if use_lgb:
        run("lgb", {"model": "lightgbm"}, budget=0.4 * (remaining() - reserve()), force=True)
    elif use_cat:
        run("catboost", {"model": "catboost"}, force=True)
    if use_xgb:
        run("xgb", {"model": "xgboost"})
    if not members and use_cat and use_lgb:
        run("catboost", {"model": "catboost"})
    if not members:
        raise RuntimeError("No boosting backend completed validation")
    best = blend(None)

    # Шаг 2: сиды XGBoost и LightGBM, пока следующая модель укладывается в остаток
    task_final = next(iter(members.values()))["config"]["task_type"]
    queue = []
    if use_lgb and task_final in CLASSIFICATION:
        queue.append(("lgb_balanced", {"model": "lightgbm", "class_weight": "balanced"}))
    elif use_lgb:
        y_num = pd.to_numeric(pd.Series(np.asarray(y)), errors="coerce").dropna()
        if len(y_num) and y_num.min() >= 0 and y_num.skew() > 1:
            # Тяжелый хвост: log1p-модель как кандидат в бленд, веса решат, нужна ли она
            queue.append(("lgb_log1p", {"model": "lightgbm", "log_target": True}))

    def add_seeds(first: int, last: int) -> None:
        for k in range(first, last):
            if use_xgb:
                queue.append((f"xgb_seed{k + 1}", {"model": "xgboost", "seed": seed0 + k}))
            if use_lgb:
                queue.append((f"lgb_seed{k + 1}", {"model": "lightgbm", "seed": seed0 + k}))

    add_seeds(1, min(max_seeds, _AUTO_MAX_SEEDS))
    added = [run(name, extra) for name, extra in queue]
    if max_seeds > _AUTO_MAX_SEEDS:
        # Запас времени: подбор параметров LightGBM, медленные модели, затем еще сиды
        queue = []
        avail = remaining() - reserve()
        lgb_time = kind_time.get("lightgbm")
        if lgb_time and avail > _AUTO_TUNE_MIN_TRIALS * lgb_time:
            tuned = tune_lightgbm(X, y, X_test, task_type, metric, base,
                                  min(0.4 * avail, _AUTO_TUNE_MAX_TRIALS * lgb_time))
            if tuned:
                queue += [("lgb_tuned", {"model": "lightgbm", "params": tuned}),
                          ("lgb_tuned_seed2", {"model": "lightgbm", "params": tuned, "seed": seed0 + 1})]
        slow = {"early_stopping_rounds": 300, "num_boost_round": 10000}
        if use_xgb:
            queue.append(("xgb_slow", {"model": "xgboost", "params": {"eta": 0.02}} | slow))
        if use_lgb:
            queue.append(("lgb_slow", {"model": "lightgbm", "params": {"learning_rate": 0.02}} | slow))
        add_seeds(_AUTO_MAX_SEEDS, max_seeds)
        added += [run(name, extra) for name, extra in queue]
    if any(added):
        best = blend(best)

    # Шаг 3: CatBoost только по флагу, в самом конце и только при большом запасе
    if use_catboost and use_cat and "catboost" not in members:
        if remaining() > 0.5 * time_budget:
            if run("catboost", {"model": "catboost"}):
                best = blend(best)
        else:
            skipped_budget.append("catboost")
            logger.info("auto_train: CatBoost пропущен, осталось %.0f%% бюджета",
                        100 * max(remaining(), 0) / time_budget)

    if best is None:
        greater = next(iter(members.values()))["greater_is_better"]
        best = dict(max(members.values(), key=lambda r: r["score"] if greater else -r["score"]))
    best["members"] = {k: {"score": r["score"], "score_std": r["score_std"], "fit_time": r["fit_time"],
                           "model": r["config"]["model"]} for k, r in members.items()}
    # Полные результаты моделей, чтобы финализатор мог пересобрать бленд без переобучения
    best["member_results"] = members
    best["fit_time"] = time.monotonic() - t_start
    best["auto_stop"] = {"reason": "queue_exhausted", "completed": list(members),
                         "skipped_for_budget": skipped_budget, "incomplete": incomplete}
    logger.info("auto_train: итог %s=%.5f +- %.5f за %.0f с из %.0f, модели %s",
                best["config"]["metric"], best["score"], best["score_std"], best["fit_time"],
                time_budget, {k: round(v["score"], 5) for k, v in best["members"].items()})
    return best


# ---------------------------------------------------------------------------
# Самопроверка на синтетике
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from sklearn.datasets import make_classification, make_regression
    from sklearn.metrics import f1_score

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    log = logging.getLogger("selftest")
    rng = np.random.default_rng(0)

    def add_noise_columns(X: pd.DataFrame) -> pd.DataFrame:
        # Категориальная колонка и пропуски, чтобы проверить подготовку данных
        X["cat"] = rng.choice(["a", "b", "c", None], size=len(X))
        X.loc[rng.random(len(X)) < 0.05, "f0"] = np.nan
        return X

    def report(name, res):
        log.info("[%s] score=%.5f std=%.5f raw=%.5f time=%.1fs folds=%s",
                 name, res["score"], res["score_std"], res["score_raw"], res["fit_time"],
                 [round(s, 4) for s in res["fold_scores"]])

    # Регрессия
    Xr, yr = make_regression(n_samples=6000, n_features=20, n_informative=12, noise=10, random_state=0)
    Xr = add_noise_columns(pd.DataFrame(Xr, columns=[f"f{i}" for i in range(20)]))
    Xr_tr, Xr_te, yr_tr = Xr.iloc[:5000], Xr.iloc[5000:].reset_index(drop=True), pd.Series(yr[:5000])
    Xr_te.loc[0, "cat"] = "unseen"
    res = train_and_predict(Xr_tr, yr_tr, Xr_te, "regression", "rmse")
    assert res["predictions"].shape == (1000,) and np.isfinite(res["predictions"]).all()
    assert yr_tr.min() <= res["predictions"].min() and res["predictions"].max() <= yr_tr.max()
    report("regression rmse", res)
    res_noclip = train_and_predict(Xr_tr, yr_tr, Xr_te, "regression", "rmse", {"clip_target": False})
    report("regression rmse без клиппинга", res_noclip)
    report("blend регрессии", blend_results([res, res_noclip], yr_tr, "rmse"))

    # Регрессия с тяжелым хвостом и log1p
    y_heavy = pd.Series(np.expm1((yr[:5000] - yr.min()) / 150))
    res = train_and_predict(Xr_tr, y_heavy, Xr_te, "regression", "rmse", {"log_target": True})
    assert res["config"]["log_target"] and np.isfinite(res["predictions"]).all()
    report("regression log1p", res)

    # Мультикласс с дисбалансом и строковыми метками
    Xc, yc = make_classification(n_samples=8000, n_features=25, n_informative=10, n_classes=5,
                                 n_clusters_per_class=2, weights=[0.55, 0.25, 0.12, 0.05, 0.03],
                                 flip_y=0.03, random_state=1)
    Xc = add_noise_columns(pd.DataFrame(Xc, columns=[f"f{i}" for i in range(25)]))
    labels = np.array(["Normal", "DoS", "Exploits", "Backdoor", "Worms"])[yc]
    Xc_tr, Xc_te = Xc.iloc[:6500], Xc.iloc[6500:].reset_index(drop=True)
    yc_tr, yc_te = pd.Series(labels[:6500]), labels[6500:]
    res = train_and_predict(Xc_tr, yc_tr, Xc_te, "multiclass", "macro_f1")
    assert res["probabilities"].shape == (1500, 5) and len(res["class_order"]) == 5
    assert set(res["predictions"]) <= set(res["class_order"])
    report("multiclass macro_f1", res)
    raw_holdout = f1_score(yc_te, np.array(res["class_order"])[res["probabilities"].argmax(1)], average="macro")
    log.info("  holdout macro_f1: без весов %.5f, с весами %.5f, веса %s", raw_holdout,
             f1_score(yc_te, res["predictions"], average="macro"), res["postprocess"]["weights"])
    # Проверка совпадения быстрой метрики с sklearn
    enc = LabelEncoder().fit(labels)
    assert abs(_macro_f1(enc.transform(yc_te), enc.transform(res["predictions"]))
               - f1_score(yc_te, res["predictions"], average="macro")) < 1e-12

    # Бленд двух сидов модели на одинаковых фолдах
    res2 = train_and_predict(Xc_tr, yc_tr, Xc_te, "multiclass", "macro_f1", {"seed": 7, "fold_seed": 42})
    blend = blend_results([res, res2], yc_tr, "macro_f1")
    assert blend["probabilities"].shape == (1500, 5) and set(blend["predictions"]) <= set(labels)
    report("blend 2 сида", blend)
    log.info("  holdout macro_f1 бленда %.5f", f1_score(yc_te, blend["predictions"], average="macro"))
    assert abs(_logloss(enc.transform(yc_te), res["probabilities"])
               - log_loss(enc.transform(yc_te), res["probabilities"], labels=np.arange(5))) < 1e-9

    # Сравнение по фолдам: результат сам с собой не принимается
    cmp = compare_results(res, res2)
    log.info("  compare default -> seed7: %s", cmp)
    assert not compare_results(res, res)["accept"]

    # Взвешенный бленд с явно слабой моделью: ее вес должен быть мал
    res_weak = train_and_predict(Xc_tr[["f0", "f1"]], yc_tr, Xc_te[["f0", "f1"]], "multiclass", "macro_f1")
    wblend = optimize_blend_weights([res, res2, res_weak], yc_tr, "macro_f1")
    report("weighted blend", wblend)
    if wblend["config"].get("blend_type") == "weighted":
        log.info("  веса бленда %s", wblend["config"]["blend_weights"])
        assert wblend["config"]["blend_weights"][2] <= 0.2
    log.info("  holdout macro_f1 взвешенного бленда %.5f",
             f1_score(yc_te, wblend["predictions"], average="macro"))

    # auto_train на мультиклассе и регрессии с маленьким бюджетом
    auto = auto_train(Xc_tr, yc_tr, Xc_te, "multiclass", "macro_f1", time_budget=60)
    assert auto["probabilities"].shape == (1500, 5) and auto["fit_time"] < 75
    report("auto_train multiclass 60s", auto)
    log.info("  holdout macro_f1 auto_train %.5f", f1_score(yc_te, auto["predictions"], average="macro"))
    auto = auto_train(Xr_tr, yr_tr, Xr_te, "regression", "rmse", time_budget=40)
    assert auto["predictions"].shape == (1000,) and np.isfinite(auto["predictions"]).all()
    assert auto["fit_time"] < 40 * 1.1
    report("auto_train regression 40s", auto)

    # Расширенный режим: подбор параметров, медленные модели, больше сидов, бюджет соблюдается
    auto = auto_train(Xc_tr, yc_tr, Xc_te, "multiclass", "macro_f1", time_budget=150, max_seeds=4)
    assert auto["fit_time"] < 150 * 1.1
    report("auto_train multiclass extended 150s", auto)
    log.info("  модели: %s", list(auto["members"]))

    # Бинарная задача с roc_auc
    Xb, yb = make_classification(n_samples=6000, n_features=20, n_informative=8, weights=[0.85, 0.15],
                                 random_state=2)
    Xb = add_noise_columns(pd.DataFrame(Xb, columns=[f"f{i}" for i in range(20)]))
    yb = pd.Series(np.where(yb == 1, "yes", "no"))
    res = train_and_predict(Xb.iloc[:5000], yb[:5000], Xb.iloc[5000:], "binary", "roc_auc")
    assert res["probabilities"].shape == (1000, 2) and res["class_order"] == ["no", "yes"]
    report("binary roc_auc", res)
    log.info("  holdout roc_auc %.5f", roc_auc_score(yb[5000:] == "yes", res["probabilities"][:, 1]))

    # Бинарная с f1: подбор порога через веса
    res = train_and_predict(Xb.iloc[:5000], yb[:5000], Xb.iloc[5000:], "binary", "f1")
    report("binary f1", res)

    # Альтернативные модели и неизвестная метрика
    for model in ("xgboost", "catboost"):
        res = train_and_predict(Xc_tr, yc_tr, Xc_te, "multiclass", "macro_f1",
                                {"model": model, "n_folds": 3})
        report(f"multiclass {model}", res)
    res = train_and_predict(Xr_tr, yr_tr, Xr_te, "regression", "something_weird",
                            {"model": "catboost", "n_folds": 3})
    report("regression catboost, неизвестная метрика", res)

    # Бюджет времени: раунды должны урезаться
    res = train_and_predict(Xc_tr, yc_tr, Xc_te, "multiclass", "macro_f1",
                            {"params": {"learning_rate": 0.01}}, time_budget=5)
    assert res["fit_time"] < 15
    report("multiclass budget=5s", res)

    log.info("Все проверки пройдены")
