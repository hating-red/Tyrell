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

import lightgbm as lgb
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
}

_ROUND_KEYS = ("n_estimators", "num_iterations", "num_boost_round", "iterations", "num_rounds")
_MIN_ROUNDS = 20                  # меньше этого по бюджету времени не урезаем


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
    if cfg["n_jobs"] is None:
        cfg["n_jobs"] = min(8, os.cpu_count() or 1)
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
                d = pd.to_datetime(df[col], errors="coerce")
                df[col] = np.where(d.isna(), np.nan, d.astype("int64") / 1e9)
        elif _is_categorical(s) and cat_encoding == "frequency":
            freq = s.value_counts(normalize=True)
            Xp[col] = s.map(freq).astype(float)
            t = Xtp[col]
            Xtp[col] = t.map(freq).astype(float).where(t.isna() | t.isin(freq.index), 0.0)
        elif _is_categorical(s):
            cat = Xp[col].astype("category")
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
        params.update(objective=objective, metric="l1" if objective == "l1" else "l2")
    elif task_type == "binary":
        params.update(objective="binary", metric="binary_logloss")
    else:
        params.update(objective="multiclass", num_class=n_classes, metric="multi_logloss")
    params.update(cfg["params"])
    return params


def _fit_lightgbm(Xtr, ytr, wtr, Xva, yva, Xte, task_type, metric, n_classes, cfg, deadline):
    params = _lgb_params(task_type, metric, n_classes, cfg)
    dtrain = lgb.Dataset(Xtr, ytr, weight=wtr)
    dvalid = lgb.Dataset(Xva, yva, reference=dtrain)
    state = {"time_stopped": False, "best_iter": 0, "best": None}

    def time_callback(env):
        # Сами отслеживаем лучшую итерацию, чтобы при остановке по времени не потерять ее
        _, _, value, higher = env.evaluation_result_list[0][:4]
        if state["best"] is None or _is_better(value, state["best"], higher):
            state["best"], state["best_iter"] = value, env.iteration
        if env.iteration + 1 >= _MIN_ROUNDS and time.monotonic() > deadline:
            state["time_stopped"] = True
            raise lgb.callback.EarlyStopException(state["best_iter"], env.evaluation_result_list)
    time_callback.order = 40

    booster = lgb.train(
        params, dtrain, num_boost_round=cfg["num_boost_round"], valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(cfg["early_stopping_rounds"], verbose=False), time_callback],
    )
    best = booster.best_iteration or booster.current_iteration()
    pva = booster.predict(Xva, num_iteration=best)
    pte = booster.predict(Xte, num_iteration=best)
    if task_type == "binary":
        pva, pte = np.column_stack([1 - pva, pva]), np.column_stack([1 - pte, pte])
    return pva, pte, best, state["time_stopped"]


def _fit_xgboost(Xtr, ytr, wtr, Xva, yva, Xte, task_type, metric, n_classes, cfg, deadline):
    import xgboost as xgb

    params = {
        "eta": 0.05, "max_depth": 7, "subsample": 0.8, "colsample_bytree": 0.8,
        "min_child_weight": 1.0, "lambda": 1.0, "tree_method": "hist", "max_bin": 256,
        "nthread": cfg["n_jobs"], "seed": cfg["seed"], "verbosity": 0,
    }
    if task_type == "regression":
        if metric.name == "mae":
            params.update(objective="reg:absoluteerror", eval_metric="mae")
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
            if epoch + 1 >= _MIN_ROUNDS and time.monotonic() > deadline:
                self.stopped = True
                return True
            return False

    timer = TimeCallback()
    dtrain = xgb.DMatrix(Xtr, ytr, weight=wtr, enable_categorical=True, nthread=cfg["n_jobs"])
    dvalid = xgb.DMatrix(Xva, yva, enable_categorical=True, nthread=cfg["n_jobs"])
    booster = xgb.train(
        params, dtrain, num_boost_round=cfg["num_boost_round"], evals=[(dvalid, "valid")],
        callbacks=[xgb.callback.EarlyStopping(cfg["early_stopping_rounds"]), timer],
        verbose_eval=False,
    )
    try:
        best = booster.best_iteration + 1
    except AttributeError:
        best = booster.num_boosted_rounds()
    dtest = xgb.DMatrix(Xte, enable_categorical=True, nthread=cfg["n_jobs"])
    pva = booster.predict(dvalid, iteration_range=(0, best))
    pte = booster.predict(dtest, iteration_range=(0, best))
    if task_type == "binary":
        pva, pte = np.column_stack([1 - pva, pva]), np.column_stack([1 - pte, pte])
    return pva, pte, best, timer.stopped


def _fit_catboost(Xtr, ytr, wtr, Xva, yva, Xte, task_type, metric, n_classes, cfg, deadline,
                  cat_cols=()):
    from catboost import CatBoost, Pool

    params = {
        # one_hot_max_size: низкокардинальные категории в one-hot, CTR на них в разы медленнее
        "learning_rate": 0.08, "depth": 6, "l2_leaf_reg": 3.0, "one_hot_max_size": 10,
        "iterations": cfg["num_boost_round"], "early_stopping_rounds": cfg["early_stopping_rounds"],
        "thread_count": cfg["n_jobs"], "random_seed": cfg["seed"],
        "verbose": False, "allow_writing_files": False, "use_best_model": True,
    }
    if task_type == "regression":
        params["loss_function"] = "MAE" if metric.name == "mae" else "RMSE"
    elif task_type == "binary":
        params.update(loss_function="Logloss", class_names=[0, 1])
    else:
        params.update(loss_function="MultiClass", class_names=list(range(n_classes)))
    params.update(cfg["params"])

    class TimeCallback:
        stopped = False

        def after_iteration(self, info):
            if info.iteration + 1 >= _MIN_ROUNDS and time.monotonic() > deadline:
                self.stopped = True
                return False
            return True

    timer = TimeCallback()
    cats = [c for c in cat_cols if c in Xtr.columns]
    model = CatBoost(params)
    model.fit(Pool(Xtr, ytr, weight=wtr, cat_features=cats),
              eval_set=Pool(Xva, yva, cat_features=cats), callbacks=[timer])
    kind = "RawFormulaVal" if task_type == "regression" else "Probability"
    pva = model.predict(Pool(Xva, cat_features=cats), prediction_type=kind)
    pte = model.predict(Pool(Xte, cat_features=cats), prediction_type=kind)
    return np.asarray(pva), np.asarray(pte), int(model.get_best_iteration() or 0) + 1, timer.stopped


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
    deadline = t_start + time_budget if time_budget else float("inf")

    task_type = str(task_type).strip().lower()
    if task_type not in TASK_TYPES:
        raise ValueError(f"task_type должен быть одним из {TASK_TYPES}, получено {task_type!r}")
    if len(X) != len(y):
        raise ValueError(f"Размеры X ({len(X)}) и y ({len(y)}) не совпадают")
    if len(X_test) == 0:
        raise ValueError("X_test пустой")

    cfg = _resolve_config(config)
    cfg["model"] = _check_backend(cfg["model"])

    # Строки без таргета не участвуют в обучении, в OOF у них будет NaN
    y_series = pd.Series(np.asarray(y))
    labeled = y_series.notna().to_numpy()
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
    if log_target and y_fit.min() < 0:
        logger.warning("В таргете есть отрицательные значения, log1p отключен")
        log_target = False
    cfg["log_target"] = bool(log_target)
    y_train_scale = np.log1p(y_fit) if log_target else y_fit

    Xp, Xtp, cat_cols = _prepare_features(X, X_test, cfg["features"], cfg["cat_encoding"])
    cfg["features"] = list(Xp.columns)
    groups = None
    if cfg["split"]:
        split_col = cfg["split"].get("column")
        if cfg["split"].get("type") != "group" or split_col not in X.columns:
            raise ValueError(f"Некорректный split: {cfg['split']}, нужна колонка из X")
        groups = X[split_col].to_numpy()[labeled]
    if not labeled.all():
        Xp = Xp.iloc[np.flatnonzero(labeled)]
    if cfg["model"] == "catboost":
        Xp, Xtp = _to_catboost_frame(Xp, cat_cols), _to_catboost_frame(Xtp, cat_cols)

    # Веса объектов
    sample_weight = None
    if cfg["class_weight"] == "balanced" and task_type in CLASSIFICATION:
        counts = np.bincount(y_fit, minlength=n_classes)
        sample_weight = (len(y_fit) / (n_classes * counts))[y_fit]
    elif cfg["class_weight"] not in (None, "balanced"):
        logger.warning("class_weight=%r не поддерживается, игнорирую", cfg["class_weight"])

    folds = _make_folds(y_fit, task_type, cfg["n_folds"], cfg["fold_seed"], groups)
    clip = task_type == "regression" and bool(cfg["clip_target"])
    cfg["n_folds"] = len(folds)
    fit_fn = _BACKENDS[cfg["model"]]
    extra = {"cat_cols": cat_cols} if cfg["model"] == "catboost" else {}

    logger.info("Старт: %s, %s, метрика %s, модель %s, %d фолдов, %d строк, %d признаков (%d кат.)",
                task_type, "log1p" if log_target else "raw", metric_obj.name, cfg["model"],
                len(folds), len(y_fit), Xp.shape[1], len(cat_cols))

    n = len(y_fit)
    oof = np.full((n, n_classes), np.nan) if n_classes else np.full(n, np.nan)
    test_sum = np.zeros((len(Xtp), n_classes)) if n_classes else np.zeros(len(Xtp))
    fold_scores_raw, fold_scores, fold_times, best_iters, done_folds = [], [], [], [], []
    time_stopped_any = False

    for i, (tr_idx, va_idx) in enumerate(folds):
        now = time.monotonic()
        if i > 0 and now > deadline:
            logger.warning("Бюджет времени исчерпан, выполнено %d из %d фолдов", i, len(folds))
            break
        # Равная доля оставшегося бюджета на фолд, 10%% в запасе на постобработку
        fold_deadline = now + 0.9 * (deadline - now) / (len(folds) - i)
        try:
            pva, pte, best_iter, stopped = fit_fn(
                Xp.iloc[tr_idx], y_train_scale[tr_idx],
                None if sample_weight is None else sample_weight[tr_idx],
                Xp.iloc[va_idx], y_train_scale[va_idx], Xtp,
                task_type, metric_obj, n_classes, cfg, fold_deadline, **extra,
            )
        except Exception as e:
            raise RuntimeError(f"Ошибка обучения {cfg['model']} на фолде {i + 1}: "
                               f"{type(e).__name__}: {e}") from e
        if log_target:
            pva, pte = np.expm1(pva), np.expm1(pte)
        fold_scores_raw.append(_score_fold(y_fit[va_idx], pva, metric_obj))
        if clip:
            # Диапазон только по train фолда, чтобы скор оставался честным
            fold_scores.append(_score_fold(
                y_fit[va_idx], np.clip(pva, y_fit[tr_idx].min(), y_fit[tr_idx].max()), metric_obj))
        else:
            fold_scores.append(fold_scores_raw[-1])
        oof[va_idx] = pva
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
    t_post = time.monotonic()
    test_pred = test_sum / len(done_folds)
    score_raw = float(np.nanmean(fold_scores_raw))
    postprocess_info = None
    weights = np.ones(n_classes) if n_classes else None

    # Клиппинг регрессии оставляем, только если он улучшает скор по фолдам
    if clip:
        score_clip = float(np.nanmean(fold_scores))
        clip = _is_better(score_clip, score_raw, metric_obj.greater_is_better)
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
    if (task_type in CLASSIFICATION and not metric_obj.needs_proba and cfg["postprocess"]
            and len(done_folds) >= 2):
        weights, cross_scores, postprocess_info = _cross_fit_class_weights(
            oof, y_fit, [folds[i] for i in done_folds], metric_obj, score_raw)
        postprocess_info["weights"] = dict(zip(
            encoder.classes_.tolist(), postprocess_info.pop("weights_raw").round(4).tolist()))
        if cross_scores is not None:
            fold_scores = cross_scores

    # Финальные предсказания
    if task_type == "regression":
        predictions = _finalize_regression(test_pred, y_fit, clip)
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

    fit_time = time.monotonic() - t_start
    score = float(np.nanmean(fold_scores))
    score_std = float(np.nanstd(fold_scores))
    cfg.update(task_type=task_type, metric=metric_obj.name, n_folds_done=len(done_folds),
               time_budget=time_budget)
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
        "fold_scores": [float(s) for s in fold_scores],
        "fold_scores_raw": [float(s) for s in fold_scores_raw],
        "fold_times": fold_times,
        "best_iterations": best_iters,
        "greater_is_better": metric_obj.greater_is_better,
        "postprocess": postprocess_info,
    }


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
    if "fold_ids" in results[0]:
        fold_ids = np.asarray(results[0]["fold_ids"])
        if any("fold_ids" in r and not np.array_equal(r["fold_ids"], fold_ids) for r in results):
            logger.warning("У результатов разные фолды, оценка бленда идет по фолдам первого результата")
        fid = fold_ids[labeled]
        folds = [(np.flatnonzero((fid >= 0) & (fid != k)), np.flatnonzero(fid == k))
                 for k in np.unique(fid[fid >= 0])]
    else:
        folds = _make_folds(y_enc, task_type, cfg0["n_folds"], cfg0.get("fold_seed", cfg0["seed"]))
    finite = np.isfinite(oofs).reshape(len(results), len(y_enc), -1).all(axis=(0, 2))
    folds = [f for f in folds if finite[f[1]].all()]
    if not folds:
        raise ValueError("Нет фолдов с полными OOF у всех моделей")
    return _BlendData(task_type, metric_obj, labeled, y_enc, class_order, oofs, tests, folds, cfg0)


def _finish_blend(data: _BlendData, oof: np.ndarray, test_pred: np.ndarray, config: dict,
                  t_start: float, members_fit_time: float) -> dict:
    """Скор по фолдам, постобработка и итоговый словарь в формате train_and_predict."""
    metric_obj, y_enc, folds = data.metric, data.y_enc, data.folds
    oof = oof.copy()
    fold_scores_raw = [_score_fold(y_enc[va], oof[va], metric_obj) for _, va in folds]
    score_raw = float(np.nanmean(fold_scores_raw))
    fold_scores = fold_scores_raw
    postprocess_info = None

    if data.task_type == "regression":
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
        if not metric_obj.needs_proba and len(folds) >= 2:
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
    config = config | {"task_type": data.task_type, "metric": metric_obj.name,
                       "n_folds": cfg0["n_folds"], "n_folds_done": len(folds), "seed": cfg0["seed"],
                       "fold_seed": cfg0.get("fold_seed", cfg0["seed"])}
    score, score_std = float(np.nanmean(fold_scores)), float(np.nanstd(fold_scores))
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
    for _, va in data.folds:
        fit_mask = done_mask.copy()
        fit_mask[va] = False
        fit_idx = np.flatnonzero(fit_mask)
        w = _greedy_blend_weights(data.oofs[:, fit_idx], data.y_enc[fit_idx], objective)
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
    if not _is_better(res["score"], best_single["score"], greater):
        logger.info("Бленд не лучше лучшей одиночной модели, возвращаю ее")
        return best_single | {"blend_rejected_score": res["score"]}
    return res


# ---------------------------------------------------------------------------
# Сравнение результатов: правило принятия гипотез
# ---------------------------------------------------------------------------

def compare_results(a: dict, b: dict) -> dict:
    """Попарное сравнение b с a по одинаковым фолдам.

    diffs положительны, когда b лучше. accept=True, только если b лучше на большинстве фолдов
    и средняя разница больше половины std разниц.
    """
    fa, fb = np.asarray(a["fold_scores"], float), np.asarray(b["fold_scores"], float)
    if len(fa) != len(fb):
        raise ValueError(f"Разное число фолдов: {len(fa)} и {len(fb)}, сравнение некорректно")
    ca, cb = a.get("config", {}), b.get("config", {})
    if ca.get("fold_seed", ca.get("seed")) != cb.get("fold_seed", cb.get("seed")):
        logger.warning("У результатов разные сиды фолдов, попарное сравнение неточно")
    greater = a.get("greater_is_better", True)
    diffs = fb - fa if greater else fa - fb
    n_better = int((diffs > 0).sum())
    diff_mean = float(diffs.mean())
    diff_std = float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0
    accept = n_better > len(diffs) / 2 and diff_mean > 0.5 * diff_std and diff_mean > 0
    return {"accept": bool(accept), "diff_mean": diff_mean, "diff_std": diff_std,
            "n_better": n_better, "n_folds": len(diffs), "diffs": diffs.round(6).tolist()}


# ---------------------------------------------------------------------------
# Автоматический рецепт
# ---------------------------------------------------------------------------

_AUTO_MIN_STEP = 20.0             # меньше стольких секунд на модель не запускаем (не больше 10% бюджета)
_AUTO_MAX_SEEDS = 3               # максимум моделей одного типа с разными сидами
_AUTO_SAFETY = 1.3                # запас к оценке времени следующей модели
_AUTO_UNSEEN_MAX = 0.01           # при большей доле новых категорий в test XGBoost не используем
_AUTO_TUNE_MIN_TRIALS = 8         # подбор параметров, только если влезает столько обучений LightGBM
_AUTO_TUNE_MAX_TRIALS = 40        # и не дольше стольких обучений


def tune_lightgbm(X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame, task_type: str, metric: str,
                  config: dict, time_budget: float) -> dict:
    """Подбор гиперпараметров LightGBM через Optuna на тех же фолдах, что и остальные модели.

    Во время подбора постобработка выключена ради скорости, ранжирование от этого почти
    не меняется. Первой пробуется точка с дефолтами, поэтому результат не хуже базового.
    Возвращает лучшие params или пустой словарь, если ни одна попытка не завершилась.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    deadline = time.monotonic() + time_budget
    greater = get_metric(metric, task_type).greater_is_better
    base = dict(config) | {"model": "lightgbm", "postprocess": False}

    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 200, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 30.0, log=True),
        }
        left = deadline - time.monotonic()
        if left < 10:
            trial.study.stop()
            raise optuna.TrialPruned()
        res = train_and_predict(X, y, X_test, task_type, metric, base | {"params": params},
                                time_budget=left)
        # Фолды, урезанные по времени, дают несравнимый скор
        if res["config"]["n_folds_done"] < res["config"]["n_folds"]:
            raise optuna.TrialPruned()
        return res["score"]

    study = optuna.create_study(direction="maximize" if greater else "minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.enqueue_trial({"learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 20,
                         "feature_fraction": 0.8, "bagging_fraction": 0.8, "lambda_l2": 1.0})
    t0 = time.monotonic()
    study.optimize(objective, timeout=time_budget, catch=(Exception,))
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not done:
        logger.warning("Подбор параметров: ни одна попытка не завершилась")
        return {}
    logger.info("Подбор параметров LightGBM: %d попыток за %.0f с, лучший %s=%.5f, параметры %s",
                len(done), time.monotonic() - t0, metric, study.best_value,
                {k: round(v, 4) if isinstance(v, float) else v for k, v in study.best_params.items()})
    return dict(study.best_params) | {"bagging_freq": 1}


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
                logger.info("auto_train: %s пропущен, доступно %.0f с", name, budget)
                return False
            if est is not None and est * _AUTO_SAFETY > budget:
                logger.info("auto_train: %s пропущен, оценка %.0f с больше доступных %.0f с",
                            name, est * _AUTO_SAFETY, budget)
                return False
        cfg = base | extra | {"params": base.get("params", {}) | extra.get("params", {})}
        logger.info("auto_train: %s, бюджет %.0f с, осталось %.0f с", name, budget, remaining())
        try:
            res = train_and_predict(X, y, X_test, task_type, metric, cfg, time_budget=budget)
        except Exception:
            logger.exception("auto_train: модель %s упала, продолжаю без нее", name)
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
    use_xgb = max(unseen.values(), default=0.0) < _AUTO_UNSEEN_MAX
    if not use_xgb:
        logger.warning("auto_train: в test новые категории %s, XGBoost не используем",
                       {k: round(v, 3) for k, v in unseen.items() if v >= _AUTO_UNSEEN_MAX})

    # Шаг 1: два бустинга с дефолтами, LightGBM первым как самый быстрый
    run("lgb", {"model": "lightgbm"}, budget=0.4 * (remaining() - reserve()), force=True)
    if not members:
        raise RuntimeError("auto_train: базовая модель не обучилась")
    if use_xgb:
        run("xgb", {"model": "xgboost"})
    best = blend(None)

    # Шаг 2: сиды XGBoost и LightGBM, пока следующая модель укладывается в остаток
    task_final = next(iter(members.values()))["config"]["task_type"]
    queue = []
    if task_final in CLASSIFICATION:
        queue.append(("lgb_balanced", {"model": "lightgbm", "class_weight": "balanced"}))
    else:
        y_num = pd.to_numeric(pd.Series(np.asarray(y)), errors="coerce").dropna()
        if len(y_num) and y_num.min() >= 0 and y_num.skew() > 1:
            # Тяжелый хвост: log1p-модель как кандидат в бленд, веса решат, нужна ли она
            queue.append(("lgb_log1p", {"model": "lightgbm", "log_target": True}))

    def add_seeds(first: int, last: int) -> None:
        for k in range(first, last):
            if use_xgb:
                queue.append((f"xgb_seed{k + 1}", {"model": "xgboost", "seed": seed0 + k}))
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
        queue.append(("lgb_slow", {"model": "lightgbm", "params": {"learning_rate": 0.02}} | slow))
        add_seeds(_AUTO_MAX_SEEDS, max_seeds)
        added += [run(name, extra) for name, extra in queue]
    if any(added):
        best = blend(best)

    # Шаг 3: CatBoost только по флагу, в самом конце и только при большом запасе
    if use_catboost:
        if remaining() > 0.5 * time_budget:
            if run("catboost", {"model": "catboost"}):
                best = blend(best)
        else:
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
