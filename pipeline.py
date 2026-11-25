import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from data_tools import detect_split, label_available_at, load_data, parse_event_datetime
from publication import MODEL_SUCCESS, TRAINED_FALLBACK
from submission import make_submission, validate_submission

PREDS_DIR = ".tyrell_preds"       # OOF и предсказания test завершенных экспериментов для ансамбля

# Ключи конфига, которые имеют смысл для всего рецепта auto, а не для одной модели
AUTO_KEYS = ("features", "cat_encoding", "target_encoding", "refit_full", "params", "model", "entity_history", "categorical_ids")


def apply_feature_code(feature_code, X_train, X_test):
    """Применяет код признаков от агента одинаково к train и test.

    Код должен определять make_features(df) -> DataFrame с теми же строками. В df нет таргета,
    поэтому утечки через него не будет. train и test идут одной таблицей, чтобы частоты и
    агрегаты считались одинаково.
    """
    namespace = {"pd": pd, "np": np}
    exec(feature_code, namespace)
    make_features = namespace.get("make_features")
    if not callable(make_features):
        raise ValueError("feature_code должен определять функцию make_features(df).")
    full = pd.concat([X_train, X_test], ignore_index=True)
    # A fresh RangeIndex after sorting cannot prove that predictions still match test IDs.
    full.index = pd.RangeIndex(-len(full), 0)
    result = make_features(full.copy())
    if not isinstance(result, pd.DataFrame) or len(result) != len(full):
        raise ValueError("make_features должна вернуть DataFrame с тем же числом строк.")
    if not result.index.equals(full.index):
        raise ValueError("make_features must preserve the input row index and order")
    if result.columns.duplicated().any():
        raise ValueError("make_features вернула повторяющиеся названия колонок.")
    result = result.reset_index(drop=True)
    return result.iloc[:len(X_train)].reset_index(drop=True), result.iloc[len(X_train):].reset_index(drop=True)


def run_experiment(task_dir, task_spec, config=None, time_budget=300, recipe="single", feature_code=None):
    from training import METRICS, auto_train, train_and_predict

    deadline = time.monotonic() + time_budget
    task_dir = Path(task_dir)
    data = load_data(task_dir, task_spec)
    task_type, metric = task_spec["task_type"], task_spec["metric"]
    if metric not in METRICS or task_type not in METRICS[metric].task_types:
        raise ValueError("Неизвестная метрика или метрика не соответствует типу задачи.")
    split = detect_split(data["X_train"], data["X_test"], time_column=task_spec.get("time_column"))
    temporal = bool(split and split["type"] == "time")
    class_count = data["y_train"].nunique()
    if task_type in ("binary", "multiclass"):
        if class_count < 2 or (task_type == "binary" and class_count != 2):
            raise ValueError("Количество классов не соответствует task_type.")
        if not split and data["y_train"].value_counts().min() < 2:
            raise ValueError("Для стратифицированной валидации нужно хотя бы 2 строки на класс.")
    if not math.isfinite(time_budget) or time_budget <= 0:
        raise ValueError("time_budget должен быть положительным конечным числом.")
    if recipe not in ("single", "auto", "bag"):
        raise ValueError("recipe должен быть single, auto или bag.")

    history_path = task_dir / ".tyrell_experiments.json"
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
    if history and history[0]["task_spec"] != task_spec:
        raise ValueError("Не меняйте тип задачи, target, ID или метрику между экспериментами.")

    X_train, X_test = data["X_train"].reset_index(drop=True), data["X_test"].reset_index(drop=True)
    cfg = dict(config or {})
    diagnostic_group = cfg.pop("diagnostic_group", None)
    excluded = cfg.pop("exclude_columns", [])
    if not isinstance(excluded, list) or any(c not in X_train or c not in X_test for c in excluded):
        raise ValueError("exclude_columns must name features present in both frames")
    if split and split["column"] in excluded:
        raise ValueError("Cannot exclude the validation split column")
    if excluded:
        X_train, X_test = X_train.drop(columns=excluded), X_test.drop(columns=excluded)
    if temporal:
        if feature_code:
            raise ValueError("Use row-local temporal auto_features for chronological experiments")
        column = split["column"]
        if not pd.api.types.is_numeric_dtype(X_train[column]):
            for frame in (X_train, X_test):
                frame[column] = parse_event_datetime(frame[column])
    if feature_code:
        X_train, X_test = apply_feature_code(feature_code, X_train, X_test)
    y_train = data["y_train"].reset_index(drop=True)
    auto_features = bool(cfg.pop("auto_features", False))
    if auto_features:
        from features import add_auto_features
        # Column selection is independent of validation labels.
        X_train, X_test = add_auto_features(X_train, X_test, row_local=temporal)

    # Кодирование по таргету: ключи ищем по исходным колонкам, само кодирование считается
    # внутри фолдов в training.py. True - ключи найти автоматически, список - заданные колонки
    target_encoding = cfg.pop("target_encoding", False)
    te_specs = None
    if temporal:
        target_encoding = False
    if target_encoding:
        from features import add_key_columns, target_encoding_keys
        if target_encoding is True:
            exclude = [split["column"]] if split else []
            periods = data["X_train"][split["column"]] if split else None
            te_specs = target_encoding_keys(data["X_train"], data["X_test"], exclude, periods)
        else:
            items = target_encoding if isinstance(target_encoding, list) else [target_encoding]
            te_specs = {"_".join(map(str, c)) if isinstance(c, list) else str(c): c if isinstance(c, list) else [c]
                        for c in items}
            if any(col not in data["X_train"] for cols in te_specs.values() for col in cols):
                raise ValueError("target_encoding: колонки должны быть среди исходных признаков.")
        if not te_specs:
            raise ValueError("Ключей для кодирования по таргету не найдено: в train и test нет "
                             "повторяющейся сущности. Можно задать колонки списком.")
        X_train, X_test, cfg["target_encoding"] = add_key_columns(X_train, X_test, te_specs)
    if recipe == "auto":
        # Модель на всем train не меняет скор CV, поэтому включаем ее в сильном рецепте всегда
        cfg.setdefault("refit_full", True)

    for key in ("n_folds", "fold_seed", "split", "n_jobs"):
        cfg.pop(key, None)
    max_seeds = int(cfg.pop("max_seeds", None) or 3)
    cfg["params"] = dict(cfg.get("params") or {})
    for key in ("num_threads", "nthread", "thread_count", "n_jobs"):
        if key in cfg["params"]:
            cfg["params"][key] = 8
    fixed = {"n_folds": 5, "fold_seed": 42, "n_jobs": 8, "split": split}

    if recipe == "single" and not history and os.environ.get("TYRELL_DIAGNOSTIC_ABLATIONS") == "1":
        fixed["profile_categories"] = True
    if task_spec.get("label_delay_unit"):
        fixed["label_delay_unit"] = task_spec["label_delay_unit"]

    time_budget = deadline - time.monotonic()
    if time_budget <= 0:
        raise TimeoutError("Experiment budget expired during feature preparation")
    model_start = time.monotonic()
    if recipe == "auto":
        result = auto_train(
            X_train, y_train, X_test, task_type, metric, time_budget,
            config={k: cfg[k] for k in AUTO_KEYS if k in cfg} | fixed,
            max_seeds=max_seeds,
        )
    else:
        train = train_and_predict
        if recipe == "bag":
            from training import train_budget_bag
            train = train_budget_bag
        result = train(X_train, y_train, X_test, task_type, metric, config=cfg | fixed, time_budget=time_budget)
    report = {
        "training_seconds": round(time.monotonic() - model_start, 3),
        "task_spec": task_spec, "recipe": recipe, "feature_code": feature_code,
        "auto_features": auto_features, "target_encoding": target_encoding or False, "te_keys": te_specs,
        "split": split, "members": result.get("members"),
        "diagnostic_group": diagnostic_group, "exclude_columns": excluded,
        "config": {k: v for k, v in cfg.items() if k != "features"},
    }
    if recipe == "single" and not history and os.environ.get("TYRELL_DIAGNOSTIC_ABLATIONS") == "1":
        from data_tools import feature_coverage
        report["feature_coverage"] = feature_coverage(
            data["X_train"], data["X_test"], clock=(split or {}).get("column"))
    return _record(task_dir, task_spec, data, history, result, report)


def _cv_audit(data, result, fold_signature):
    """Aggregate only scored OOF rows and their actual expanding training prefixes."""
    from training import _time_values

    cfg = result["config"]
    split = cfg.get("split") or {}
    metric = cfg.get("metric")
    if split.get("type") != "time" or metric not in ("pr_auc", "mae"):
        return None
    clock = _time_values(data["X_train"][split["column"]]).reset_index(drop=True)
    target = data["y_train"].reset_index(drop=True)
    labeled = target.notna() & clock.notna()
    oof = np.asarray(result["oof"])
    fid = np.asarray(result["fold_ids"])
    windows = cfg["fold_windows"]
    used = np.unique(fid[fid >= 0])
    if len(windows) != len(used) or len(oof) != len(target):
        raise ValueError("Temporal OOF dimensions disagree with the recorded folds")
    binary = metric == "pr_auc"
    if binary:
        positive = result["class_order"][1]
        values = target.eq(positive).fillna(False).to_numpy(dtype=int)
        scores = oof[:, 1]
    else:
        values = pd.to_numeric(target).to_numpy(dtype=float, na_value=np.nan)
        scores = oof
    period = {"rows": int(labeled.sum()), "start": str(clock[labeled].min()), "end": str(clock[labeled].max())}
    if binary:
        period.update(positive_class=positive, positives=int(values[labeled].sum()),
                      prevalence=float(values[labeled].mean()))
    available_at = None
    test_cutoff = None
    if cfg.get("label_delay_unit"):
        available_at = label_available_at(clock, values, cfg["label_delay_unit"])
        test_cutoff = pd.Timestamp(cfg["prediction_cutoff"])
        period.update(prediction_cutoff=str(test_cutoff), observed_before_prediction=int((available_at < test_cutoff).sum()))
    member_hash = None
    signature = hashlib.sha256()
    if data.get("train_ids") is not None:
        identities = pd.DataFrame({"id": data["train_ids"].reset_index(drop=True).astype("string"), "time": clock})
        member_hash = pd.util.hash_pandas_object(identities, index=False).to_numpy(dtype="<u8")
    rows = []
    for number, window, model_score in zip(used, windows, result["fold_scores"], strict=True):
        valid = fid == number
        earliest = clock[valid].min()
        cutoff = pd.Timestamp(window["validation_cutoff"]) if available_at is not None else earliest
        train = labeled & (clock < cutoff)
        if available_at is not None:
            train &= available_at < min(cutoff, test_cutoff)
        if (int(train.sum()) != window["train_rows"] or int(valid.sum()) != window["validation_rows"]
                or str(clock[train].max()) != window["train_max"] or str(earliest) != window["validation_min"]
                or not labeled[valid].all() or not np.isfinite(scores[valid]).all()):
            raise ValueError("Cannot reconcile recorded temporal window with its scored OOF rows")
        actual, prediction = values[valid], scores[valid]
        row = dict(window, fold=int(number), validation_max=str(clock[valid].max()), model_score=float(model_score))
        if member_hash is not None:
            for mask in (train, valid):
                signature.update(int(mask.sum()).to_bytes(8, "little"))
                signature.update(np.sort(member_hash[mask]).tobytes())
        if binary:
            # Include every tie at the last selected score and report the resulting size.
            top_k = max(1, (len(prediction) + 9) // 10)
            top = prediction >= np.partition(prediction, len(prediction) - top_k)[-top_k]
            row.update(train_positives=int(values[train].sum()), train_prevalence=float(values[train].mean()),
                       validation_positives=int(actual.sum()), validation_prevalence=float(actual.mean()),
                       baseline_score=float(actual.mean()), brier=float(np.mean((prediction - actual) ** 2)),
                       mean_probability=float(prediction.mean()), top_decile_rows=int(top.sum()),
                       top_decile_precision=float(actual[top].mean()))
        else:
            median = float(np.median(values[train]))
            residual = prediction - actual
            absolute = np.abs(residual)
            row.update(train_median=median, baseline_score=float(np.abs(actual - median).mean()),
                       mean_signed_error=float(residual.mean()), median_absolute_error=float(np.median(absolute)),
                       p90_absolute_error=float(np.quantile(absolute, .9)))
        rows.append(row)
    return {"fold_signature": fold_signature,
            "membership_signature": signature.hexdigest() if member_hash is not None else None,
            "training_period": period, "folds": rows,
            "baseline_strategy": "train_prevalence" if binary else "train_median",
            "baseline_score": float(np.mean([r["baseline_score"] for r in rows]))}


def _record(task_dir, task_spec, data, history, result, report):
    """Сравнение с сохраненным, запись сабмита при улучшении, предсказания для ансамбля, журнал."""
    from training import compare_results

    if not math.isfinite(result["score"]):
        raise ValueError("Валидация вернула нечисловой результат.")
    n_folds = result["config"]["n_folds"]
    n_folds_done = result["config"].get("n_folds_done", n_folds)
    fold_signature = hashlib.sha256(np.asarray(result["fold_ids"], dtype="<i4").tobytes()).hexdigest()
    audit = _cv_audit(data, result, fold_signature)
    saved = [item for item in history if item["saved"]]
    previous = saved[-1] if saved else None
    complete = n_folds_done == n_folds
    better = previous is None
    comparison = None
    is_bag = report["recipe"] == "bag"
    same_folds = True
    if previous is not None:
        same_folds = previous.get("fold_signature") == fold_signature
        previous_membership = (previous.get("cv_audit") or {}).get("membership_signature")
        membership = (audit or {}).get("membership_signature")
        if previous_membership and (membership or is_bag):
            same_folds &= previous_membership == membership
        if same_folds and (complete or not is_bag):
            # Не берем «везучую» модель: новый результат должен быть лучше на большинстве
            # фолдов и с приростом больше шума, а не просто чуть выше в среднем
            comparison = compare_results(
                {"fold_scores": previous["fold_scores"], "greater_is_better": previous["greater_is_better"]},
                {"fold_scores": result["fold_scores"], "greater_is_better": result["greater_is_better"]},
            )
        better = (complete and not previous["complete"]) or bool(comparison and comparison["accept"])
    eligible = not report.get("diagnostic_group") and (not is_bag or (complete and same_folds))
    better = better and eligible
    problems = _write_submission(task_dir, task_spec, data, result) if better else []

    # Предсказания завершенных экспериментов храним на диске: в конце их можно смешать
    preds_file = None
    if complete and "fold_ids" in result and eligible:
        preds_dir = task_dir / PREDS_DIR
        preds_dir.mkdir(exist_ok=True)
        preds_file = str(preds_dir / f"{len(history)}.npz")
        test = result["probabilities"] if result["probabilities"] is not None else result["predictions"]
        saved_labels = {}
        if result.get("meta_known_from") is not None:
            saved_labels["meta_known_from"] = result["meta_known_from"]
        if result["class_order"] is not None:
            indices = pd.Index(result["class_order"]).get_indexer(result["predictions"])
            if (indices < 0).any():
                raise ValueError("Predicted label is absent from class_order")
            saved_labels["predicted_class"] = indices
        np.savez_compressed(preds_file, oof=np.asarray(result["oof"]),
                            test=np.asarray(test), fold_ids=result["fold_ids"], **saved_labels)

    report = report | {
        "fold_signature": fold_signature, "cv_audit": audit,
        "feature_candidates": result.get("feature_candidates"),
        "model_inputs": result.get("model_inputs"), "fold_diagnostics": result.get("fold_diagnostics"),
        "category_probes": result.get("category_probes"),
        "early_stopping_windows": result["config"].get("early_stopping_windows"),
        "prediction_cutoff": result["config"].get("prediction_cutoff"),
        "fold_windows": result["config"].get("fold_windows"),
        "score": result["score"], "score_std": result["score_std"], "metric": task_spec["metric"],
        "fold_scores": [round(s, 6) for s in result.get("fold_scores", [])],
        "greater_is_better": result["greater_is_better"], "complete": complete, "n_folds_done": n_folds_done,
        "saved": better and not problems, "problems": problems,
        "best_saved_score": result["score"] if better and not problems else (previous["score"] if previous else None),
        "vs_saved": comparison and {k: comparison[k] for k in ("accept", "diff_mean", "diff_std", "n_better")},
        "fit_time": result["fit_time"], "preds_file": preds_file, "class_order": result["class_order"],
    }
    history.append(report)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=task_dir,
                                         prefix=".tyrell_history_", delete=False) as stream:
            tmp = Path(stream.name)
            json.dump(history, stream, ensure_ascii=False, default=str)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, task_dir / ".tyrell_experiments.json")
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
    return report


def _output_path(task_dir):
    return Path(os.environ.get("TYRELL_OUTPUT_DIR") or task_dir) / "submission.csv"


def _nonnegative(spec):
    return spec.get("nonnegative", spec.get("target_column") == "response_time_minutes")


def _write_submission(task_dir, task_spec, data, result):
    """Сабмит по формату sample_submission: метки, вероятность одного класса или всех классов."""
    task_type = task_spec["task_type"]
    probabilities = task_spec.get("probabilities", False)
    predictions = result["predictions"]
    if task_type == "regression" and _nonnegative(task_spec):
        predictions = np.asarray(predictions, dtype=float)
        predictions = np.where(np.isfinite(predictions), np.maximum(predictions, 0), predictions)
    sample = pd.read_csv(data["sample_path"], nrows=0)
    answer_count = len(sample.columns) - 1
    if probabilities or answer_count > 1:
        predictions = result["probabilities"]
        probabilities = True
        if predictions is None:
            raise ValueError("Регрессионная модель не возвращает вероятности классов.")
        if answer_count == 1:
            classes = result["class_order"]
            if len(classes) != 2:
                raise ValueError("Одной колонки вероятностей недостаточно для нескольких классов.")
            positive = task_spec.get("positive_class", classes[1])
            if positive not in classes:
                raise ValueError("positive_class отсутствует среди меток обучения.")
            predictions = predictions[:, classes.index(positive)]
    labels = set(result["class_order"]) if task_type != "regression" and not probabilities else None
    return make_submission(
        predictions, data["test_ids"], data["sample_path"], _output_path(task_dir),
        allowed_labels=labels, probabilities=probabilities, class_order=result["class_order"],
        publication_state=MODEL_SUCCESS, score=float(result["score"]),
    )


def run_ensemble(task_dir, task_spec, max_members=8):
    """Смесь лучших завершенных экспериментов: разные наборы признаков ошибаются по-разному.

    Фолды у всех экспериментов одинаковые, поэтому OOF смешиваются честно; веса подбираются
    с кросс-фиттингом, а сабмит заменяется по тому же правилу, что и после эксперимента.
    """
    from training import optimize_blend_weights

    task_dir = Path(task_dir)
    history_path = task_dir / ".tyrell_experiments.json"
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
    candidates = [h for h in history if h.get("preds_file") and Path(h["preds_file"]).exists()
                  and h["recipe"] != "ensemble"]
    if len(candidates) < 2:
        raise ValueError("Для ансамбля нужно хотя бы два завершенных эксперимента.")
    memberships = {(h.get("cv_audit") or {}).get("membership_signature") for h in candidates}
    if len(memberships - {None}) > 1:
        raise ValueError("Ensemble members have different chronological fold memberships")
    if any(h["task_spec"] != task_spec for h in candidates):
        raise ValueError("Ensemble members belong to a different task")
    greater = candidates[0]["greater_is_better"]
    candidates = sorted(candidates, key=lambda h: h["score"], reverse=greater)[:max_members]

    data = load_data(task_dir, task_spec)
    is_regression = task_spec["task_type"] == "regression"
    results = []
    for h in candidates:
        with np.load(h["preds_file"]) as stored:
            test = stored["test"].astype(float)
            predictions = test
            if not is_regression:
                order = np.asarray(h["class_order"], dtype=object)
                indices = stored["predicted_class"] if "predicted_class" in stored else test.argmax(1)
                predictions = order[indices]
            results.append({
                "oof": stored["oof"].astype(float), "fold_ids": stored["fold_ids"],
                "meta_known_from": stored["meta_known_from"] if "meta_known_from" in stored else None,
                "predictions": predictions, "probabilities": None if is_regression else test,
                "class_order": h["class_order"], "score": h["score"], "score_std": h["score_std"],
                "fold_scores": [r["model_score"] for r in h["cv_audit"]["folds"]] if h.get("cv_audit") else h["fold_scores"],
                "greater_is_better": greater, "fit_time": h["fit_time"],
                "config": {"task_type": task_spec["task_type"], "metric": task_spec["metric"],
                           "n_folds": h["n_folds_done"],
                           "early_stopping_windows": h.get("early_stopping_windows"),
                           "label_delay_unit": task_spec.get("label_delay_unit"),
                           "prediction_cutoff": h.get("prediction_cutoff"),
                           "n_folds_done": h["n_folds_done"], "split": h.get("split"),
                           "fold_windows": h.get("fold_windows"), "seed": 42, "fold_seed": 42},
            })
    result = optimize_blend_weights(results, data["y_train"].reset_index(drop=True), task_spec["metric"])
    result.setdefault("config", {})
    result["config"] = result["config"] | {"n_folds": candidates[0]["n_folds_done"]}
    report = {"task_spec": task_spec, "recipe": "ensemble", "split": candidates[0].get("split"), "feature_code": None,
              "members": {f"exp{history.index(h)}": {"score": h["score"], "recipe": h["recipe"]} for h in candidates},
              "config": {"blend_weights": result["config"].get("blend_weights"),
                         "blend_type": result["config"].get("blend_type")}}
    return _record(task_dir, task_spec, data, history, result, report)


def write_fallback(task_dir, task_spec):
    """Всегда публикует прогноз текущего запуска по доступным обучающим меткам."""
    task_dir = Path(task_dir)
    data = load_data(task_dir, task_spec)
    y = data["y_train"].dropna()
    if task_spec["task_type"] == "regression":
        y = pd.to_numeric(y, errors="coerce")
        y = y[np.isfinite(y)]
    if task_spec.get("label_delay_unit"):
        split = detect_split(data["X_train"], data["X_test"], time_column=task_spec.get("time_column"))
        if not split or split["type"] != "time":
            raise ValueError("A duration target needs an event clock for the early submission")
        col = split["column"]
        train_time = parse_event_datetime(data["X_train"].loc[y.index, col])
        test_time = parse_event_datetime(data["X_test"][col])
        available = label_available_at(train_time, y, task_spec["label_delay_unit"])
        y = y[available < test_time.min()] if test_time.notna().all() else y.iloc[:0]
    if y.empty:
        raise ValueError("No eligible training labels to calculate a fallback prediction")
    sample = pd.read_csv(data["sample_path"], nrows=0)
    n = len(data["test_ids"])
    if task_spec["task_type"] == "regression":
        value = float(y.median() if task_spec["metric"] == "mae" else y.mean())
        if not np.isfinite(value):
            raise ValueError("Non-finite training statistic for the fallback prediction")
        predictions = np.full(n, max(0., value) if _nonnegative(task_spec) else value)
        probabilities, class_order, labels = False, None, None
    else:
        shares = y.value_counts(normalize=True).sort_index()
        class_order = shares.index.tolist()
        if task_spec.get("probabilities") or len(sample.columns) > 2:
            probabilities = True
            if len(sample.columns) == 2:
                default = class_order[-1]
                if len(class_order) == 1 and str(default) in {"0", "1", "-1"}:
                    default = "1" if isinstance(default, str) else 1
                positive = task_spec.get("positive_class", default)
                predictions = np.full(n, shares.get(positive, 0.0), dtype=float)
            else:
                answers = [c for c in sample.columns if c != data["id_column"]]
                by_label = {str(label): share for label, share in shares.items()}
                if not set(by_label).issubset(answers):
                    return ["Training classes are absent from the probability template"]
                predictions = pd.DataFrame({col: np.full(n, by_label.get(col, 0.0)) for col in answers})
            labels = None
        else:
            probabilities, labels = False, set(class_order)
            predictions = np.full(n, shares.idxmax(), dtype=object)
    return make_submission(
        predictions, data["test_ids"], data["sample_path"], _output_path(task_dir),
        allowed_labels=labels, probabilities=probabilities, class_order=class_order,
        publication_state=TRAINED_FALLBACK,
    )


def check_result(task_dir, task_spec=None):
    task_dir = Path(task_dir)
    if task_spec is None:
        history = json.loads((task_dir / ".tyrell_experiments.json").read_text(encoding="utf-8"))
        saved = [item for item in history if item["saved"]]
        if not saved:
            raise RuntimeError("No saved task specification")
        task_spec = saved[-1]["task_spec"]
    data = load_data(task_dir, task_spec)
    identifier = data["id_column"]
    sample = pd.read_csv(data["sample_path"], dtype={identifier: str}, keep_default_na=False)
    frame = pd.read_csv(_output_path(task_dir), dtype={identifier: str}, keep_default_na=False)
    expected = sample[identifier]
    test_ids = data["test_ids"]
    if (not expected.is_unique or not test_ids.is_unique or expected.eq("").any()
            or set(expected) != set(test_ids)):
        raise RuntimeError("IDs in sample_submission and test must match and be unique")
    probabilities = task_spec.get("probabilities", False) or len(sample.columns) > 2
    regression = task_spec["task_type"] == "regression"
    labels = set(data["y_train"].dropna()) if not regression and not probabilities else None
    problems = validate_submission(frame, expected, sample.columns.tolist(), labels, probabilities)
    if regression and not problems:
        values = pd.to_numeric(frame.drop(columns=identifier).iloc[:, 0], errors="coerce")
        if not np.isfinite(values).all() or (_nonnegative(task_spec) and (values < 0).any()):
            problems.append("invalid regression predictions")
    if problems:
        raise RuntimeError("Invalid submission: " + "; ".join(problems))


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2 or sys.argv[1] not in ("experiment", "ensemble"):
        raise SystemExit("Expected a pipeline operation: experiment or ensemble")

    request = json.load(sys.stdin.buffer)
    if sys.argv[1] == "experiment":
        report = run_experiment(".", **request)
        report["membership_signature"] = (report.get("cv_audit") or {}).get("membership_signature")
        fields = ("score", "score_std", "members", "saved", "complete", "fold_scores", "fold_signature",
                  "split", "greater_is_better", "fit_time", "config", "feature_candidates",
                  "membership_signature", "te_keys", "vs_saved", "best_saved_score", "n_folds_done",
                  "recipe", "metric")
    else:
        report = run_ensemble(".", request["task_spec"])
        fields = ("score", "saved", "vs_saved", "members")
    print(json.dumps({key: report.get(key) for key in fields}, ensure_ascii=False, default=str))
