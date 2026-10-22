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
from sklearn.model_selection import KFold, StratifiedKFold
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
    return float(log_loss(y_true, proba, labels=np.arange(proba.shape[1])))


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
    "seed": 42,
    "num_boost_round": 3000,
    "early_stopping_rounds": 100,
    "log_target": None,           # log1p таргета в регрессии, None - авто (только для rmsle)
    "class_weight": None,         # None или "balanced"
    "postprocess": True,          # подбор весов классов под метрику по меткам
    "n_jobs": None,               # None - min(8, число CPU)
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

def _prepare_features(X: pd.DataFrame, X_test: pd.DataFrame,
                      features: list | None) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Отбор признаков и приведение типов. Одна копия на датафрейм, исходники не меняются."""
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
        elif isinstance(s.dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(s) \
                or pd.api.types.is_string_dtype(s):
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


def _make_folds(y_enc: np.ndarray, task_type: str, n_folds: int, seed: int) -> list:
    n = len(y_enc)
    n_folds = max(2, min(int(n_folds), n))
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


def fit_class_weights(proba: np.ndarray, y: np.ndarray, metric: Metric,
                      n_passes: int = 3) -> tuple[np.ndarray, float]:
    """Покоординатный поиск множителей классов: pred = argmax(proba * w).

    Для бинарной задачи это эквивалент подбора порога.
    """
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

    Xp, Xtp, cat_cols = _prepare_features(X, X_test, cfg["features"])
    cfg["features"] = list(Xp.columns)
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

    folds = _make_folds(y_fit, task_type, cfg["n_folds"], cfg["seed"])
    cfg["n_folds"] = len(folds)
    fit_fn = _BACKENDS[cfg["model"]]
    extra = {"cat_cols": cat_cols} if cfg["model"] == "catboost" else {}

    logger.info("Старт: %s, %s, метрика %s, модель %s, %d фолдов, %d строк, %d признаков (%d кат.)",
                task_type, "log1p" if log_target else "raw", metric_obj.name, cfg["model"],
                len(folds), len(y_fit), Xp.shape[1], len(cat_cols))

    n = len(y_fit)
    oof = np.full((n, n_classes), np.nan) if n_classes else np.full(n, np.nan)
    test_sum = np.zeros((len(Xtp), n_classes)) if n_classes else np.zeros(len(Xtp))
    fold_scores_raw, fold_times, best_iters, done_folds = [], [], [], []
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
        oof[va_idx] = pva
        test_sum += pte
        time_stopped_any |= stopped
        done_folds.append(i)
        best_iters.append(int(best_iter))
        fold_times.append(time.monotonic() - now)

        fold_scores_raw.append(_score_fold(y_fit[va_idx], pva, metric_obj))
        logger.info("Фолд %d/%d: %s=%.5f, итераций %d, %.1f с%s", i + 1, len(folds), metric_obj.name,
                    fold_scores_raw[-1], best_iter, fold_times[-1],
                    " (урезано по времени)" if stopped else "")

    if time_stopped_any:
        logger.warning("Часть фолдов остановлена по бюджету времени, раунды урезаны")
    test_pred = test_sum / len(done_folds)
    done_mask = np.zeros(n, dtype=bool)
    for i in done_folds:
        done_mask[folds[i][1]] = True

    score_raw = float(np.nanmean(fold_scores_raw))
    fold_scores = list(fold_scores_raw)
    postprocess_info = None

    # Постобработка для метрик по меткам: веса классов, честная оценка через кросс-фиттинг
    if (task_type in CLASSIFICATION and not metric_obj.needs_proba and cfg["postprocess"]
            and len(done_folds) >= 2):
        cross_scores = []
        for i in done_folds:
            va_idx = folds[i][1]
            fit_mask = done_mask.copy()
            fit_mask[va_idx] = False
            w, _ = fit_class_weights(oof[fit_mask], y_fit[fit_mask], metric_obj)
            cross_scores.append(metric_obj.func(y_fit[va_idx], _apply_weights(oof[va_idx], w)))
        score_pp = float(np.mean(cross_scores))
        weights, _ = fit_class_weights(oof[done_mask], y_fit[done_mask], metric_obj)
        accepted = _is_better(score_pp, score_raw, metric_obj.greater_is_better)
        logger.info("Веса классов: %s %.5f -> %.5f (кросс-фиттинг), %s", metric_obj.name,
                    score_raw, score_pp, "применяю" if accepted else "не улучшает, отключаю")
        postprocess_info = {
            "type": "class_weights", "applied": accepted, "score_after": score_pp,
            "weights": dict(zip(encoder.classes_.tolist(), weights.round(4).tolist())),
        }
        if accepted:
            fold_scores = cross_scores
        else:
            weights = np.ones(n_classes)
    else:
        weights = np.ones(n_classes) if n_classes else None

    # Финальные предсказания
    if task_type == "regression":
        bad = ~np.isfinite(test_pred)
        if bad.any():
            logger.warning("В предсказаниях %d нечисловых значений, заменяю средним таргета", bad.sum())
            test_pred[bad] = float(np.mean(y_fit))
        predictions, probabilities, class_order = test_pred, None, None
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
        "fold_scores": [float(s) for s in fold_scores],
        "fold_scores_raw": [float(s) for s in fold_scores_raw],
        "fold_times": fold_times,
        "best_iterations": best_iters,
        "greater_is_better": metric_obj.greater_is_better,
        "postprocess": postprocess_info,
    }


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
    report("regression rmse", res)

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
