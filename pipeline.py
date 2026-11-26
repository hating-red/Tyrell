import hashlib
import json
import math
import re
import os
import sys
import shutil
import tempfile
from contextvars import ContextVar
from itertools import chain, combinations

from pathlib import Path

import numpy as np
import pandas as pd

from data_tools import detect_split, load_data, parse_event_datetime, label_available_at
from publication import MODEL_SUCCESS, TRAINED_FALLBACK, SCHEMA_FALLBACK, csv_digest, prepare_publication, receipt_directory
from submission import make_submission, validate_submission

PREDS_DIR = ".tyrell_preds"       # OOF и предсказания test завершенных экспериментов для ансамбля
output_directory = ContextVar("output_directory", default=None)

# Ключи конфига, которые имеют смысл для всего рецепта auto, а не для одной модели
AUTO_KEYS = ("features", "cat_encoding", "target_encoding", "refit_full", "entity_history", "log_target",
             "ad_history", "ad_history_window", "ad_history_smoothing")


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
    from training import METRICS, auto_train, compare_results, train_and_predict

    task_dir = Path(task_dir)
    data = load_data(task_dir, task_spec)
    task_type, metric = task_spec["task_type"], task_spec["metric"]
    if metric not in METRICS or task_type not in METRICS[metric].task_types:
        raise ValueError("Неизвестная метрика или метрика не соответствует типу задачи.")
    split = detect_split(data["X_train"], data["X_test"], time_column=task_spec.get("time_column"))
    temporal = (split or {}).get("type") == "time"
    class_count = data["y_train"].nunique()
    if task_type in ("binary", "multiclass"):
        if class_count < 2 or (task_type == "binary" and class_count != 2):
            raise ValueError("Количество классов не соответствует task_type.")
        if data["y_train"].value_counts().min() < 2:
            raise ValueError("Для стратифицированной валидации нужно хотя бы 2 строки на класс.")
    if not math.isfinite(time_budget) or time_budget <= 0:
        raise ValueError("time_budget должен быть положительным конечным числом.")
    if recipe not in ("single", "auto"):
        raise ValueError("recipe должен быть single или auto.")

    history = experiment_history(task_dir)
    if history and history[0]["task_spec"] != task_spec:
        raise ValueError("Не меняйте тип задачи, target, ID или метрику между экспериментами.")

    X_train, X_test = data["X_train"].reset_index(drop=True), data["X_test"].reset_index(drop=True)
    if temporal:
        clock = split["column"]
        if feature_code:
            raise ValueError("Temporal features must use the row-local transformation")
        if not pd.api.types.is_numeric_dtype(X_train[clock]):
            for frame in (X_train, X_test):
                frame[clock] = parse_event_datetime(frame[clock])
    if feature_code:
        X_train, X_test = apply_feature_code(feature_code, X_train, X_test)
    y_train = data["y_train"].reset_index(drop=True)
    cfg = dict(config or {})
    auto_features = bool(cfg.pop("auto_features", False))
    if auto_features:
        from features import add_auto_features, important_numeric
        priority = None if temporal else important_numeric(X_train, y_train, task_type)
        X_train, X_test = add_auto_features(X_train, X_test, priority, row_local=temporal)

    # Фолды одинаковые во всех экспериментах, иначе скоры несравнимы: агент их не меняет.
    # Разбиение группами включается само, если test похож на новый период

    # Кодирование по таргету: ключи ищем по исходным колонкам, само кодирование считается
    # внутри фолдов в training.py. True - ключи найти автоматически, список - заданные колонки
    target_encoding = cfg.pop("target_encoding", False)
    te_specs = None
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
            return {"skipped": "No repeated target-encoding key shared across periods"}
        X_train, X_test, cfg["target_encoding"] = add_key_columns(
            X_train, X_test, te_specs, raw_frames=(data["X_train"], data["X_test"]))
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

    if task_spec.get("label_delay_unit"):
        fixed["label_delay_unit"] = task_spec["label_delay_unit"]

    if recipe == "auto":
        result = auto_train(
            X_train, y_train, X_test, task_type, metric, time_budget,
            config={k: cfg[k] for k in AUTO_KEYS if k in cfg} | fixed,
            max_seeds=max_seeds, use_catboost=True,
        )
    else:
        result = train_and_predict(
            X_train, y_train, X_test, task_type, metric, config=cfg | fixed, time_budget=time_budget,
        )
    report = {
        "task_spec": task_spec, "recipe": recipe, "feature_code": feature_code,
        "auto_features": auto_features, "target_encoding": target_encoding or False, "te_keys": te_specs,
        "split": split, "members": result.get("members"),
        "config": {k: v for k, v in cfg.items() if k != "features"},
    }
    report["feature_candidates"] = result.get("feature_candidates", {})
    if "auto_stop" in result:
        report["auto_stop"] = result["auto_stop"]
    return _record(task_dir, task_spec, data, history, result, report)


def _beats_fold_baselines(scores, baselines, greater):
    if not baselines or len(scores) != len(baselines):
        return False
    return all((score > base["score"] if greater else score < base["score"])
               for score, base in zip(scores, baselines))


def _record(task_dir, task_spec, data, history, result, report):
    """Сравнение с сохраненным, запись сабмита при улучшении, предсказания для ансамбля, журнал."""
    from training import compare_results

    if not math.isfinite(result["score"]):
        raise ValueError("Валидация вернула нечисловой результат.")
    n_folds = result["config"]["n_folds"]
    n_folds_done = result["config"].get("n_folds_done", n_folds)
    signature = hashlib.sha256(np.asarray(result["fold_ids"], dtype="<i4").tobytes()).hexdigest()
    baselines = result["config"].get("fold_baselines", [])
    greater = result["greater_is_better"]
    saved = [item for item in history if item["saved"]]
    previous = saved[-1] if saved else None
    complete = n_folds_done == n_folds
    better = previous is None
    comparison = None
    if previous is not None:
        same_folds = signature == previous.get("fold_signature")
        if same_folds:
            # Не берем «везучую» модель: новый результат должен быть лучше на большинстве
            # фолдов и с приростом больше шума, а не просто чуть выше в среднем
            comparison = compare_results(
                {"fold_scores": previous["fold_scores"], "greater_is_better": previous["greater_is_better"]},
                {"fold_scores": result["fold_scores"], "greater_is_better": result["greater_is_better"]},
            )
        better = (complete and not previous["complete"]) or (same_folds and comparison["accept"])
        if same_folds and baselines and previous.get("fold_baselines"):
            incumbent_wins = _beats_fold_baselines(previous["fold_scores"], previous["fold_baselines"], greater)
            candidate_wins = _beats_fold_baselines(result["fold_scores"], baselines, greater)
            if incumbent_wins != candidate_wins:
                # Clearing every baseline is a release criterion. A new ensemble still
                # has to improve the incumbent's mean and a majority of its folds.
                blend_improves = (comparison["n_better"] > n_folds / 2 and comparison["diff_mean"] > 0)
                better = candidate_wins and complete and (report.get("recipe") != "ensemble" or blend_improves)
        if report.get("config", {}).get("ad_history"):
            better = bool(better and same_folds and comparison["n_better"] > n_folds / 2)
    # Предсказания завершенных экспериментов храним на диске: в конце их можно смешать
    preds_file = None
    if complete and "fold_ids" in result:
        preds_dir = task_dir / PREDS_DIR
        preds_dir.mkdir(exist_ok=True)
        preds_file = str(preds_dir / f"{len(history)}.npz")
        test = result["probabilities"] if result["probabilities"] is not None else result["predictions"]
        extras = {}
        if result.get("meta_known_from") is not None:
            extras["meta_known_from"] = result["meta_known_from"]
        if result["class_order"] is not None:
            extras["predicted_class"] = pd.Index(result["class_order"]).get_indexer(result["predictions"])
        np.savez_compressed(preds_file, oof=np.asarray(result["oof"], dtype=np.float32),
                            test=np.asarray(test, dtype=np.float64), fold_ids=result["fold_ids"],
                            **extras)

    output = _output_path(task_dir)
    staged_csv = task_dir / ".tyrell_candidate.csv"
    problems = _write_submission(staged_csv, task_spec, data, result) if better else []
    report = report | {
        "score": result["score"], "score_std": result["score_std"], "metric": task_spec["metric"],
        "fold_scores": [float(s) for s in result.get("fold_scores", [])],
        "fold_baselines": baselines,
        "greater_is_better": result["greater_is_better"], "complete": complete, "n_folds_done": n_folds_done,
        "saved": better and not problems, "problems": problems,
        "best_saved_score": result["score"] if better and not problems else (previous["score"] if previous else None),
        "vs_saved": comparison and {k: comparison[k] for k in ("accept", "diff_mean", "diff_std", "n_better")},
        "fold_signature": signature, "fold_windows": result["config"].get("fold_windows", []),
        "fit_time": result["fit_time"], "preds_file": preds_file, "class_order": result["class_order"],
    }
    history_file = task_dir / ".tyrell_experiments.json"
    staged_history = history_file.with_suffix(".tmp")
    if report["saved"]:
        with staged_csv.open("rb") as stream:
            report["submission_sha256"] = csv_digest(stream)
    staged_history.write_text(json.dumps([*history, report], ensure_ascii=False, default=str), encoding="utf-8")
    if report["saved"]:
        # Both files are ready. A worker interrupted between these two renames
        # leaves enough information for the next trial to recover the incumbent.
        pending = history_file.with_suffix(".pending")
        staged_history.replace(pending)
        if staged_csv.stat().st_dev == output.parent.stat().st_dev:
            staged_csv.replace(output)
        else:
            with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as tmp:
                copied = Path(tmp.name)
            try:
                shutil.copyfile(staged_csv, copied)
                copied.chmod(0o644)
                prepare_publication(copied, MODEL_SUCCESS, result["score"])
                copied.replace(output)
            finally:
                copied.unlink(missing_ok=True)
        pending.replace(history_file)
    else:
        staged_history.replace(history_file)
    history.append(report)
    return report


def _output_path(task_dir):
    return Path(output_directory.get() or os.environ.get("TYRELL_OUTPUT_DIR", task_dir)) / "submission.csv"


def experiment_history(task_dir):
    history_file = Path(task_dir) / ".tyrell_experiments.json"
    pending = history_file.with_suffix(".pending")
    if pending.exists():
        proposed = json.loads(pending.read_text(encoding="utf-8"))
        output = _output_path(task_dir)
        if output.is_file():
            with output.open("rb") as stream:
                digest = csv_digest(stream)
            if proposed[-1]["submission_sha256"] == digest:
                pending.replace(history_file)
        pending.unlink(missing_ok=True)
    return json.loads(history_file.read_text(encoding="utf-8")) if history_file.exists() else []


def _nonnegative(spec):
    return bool(spec.get("nonnegative"))


def _write_submission(output_path, task_spec, data, result):
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
        predictions, data["test_ids"], data["sample_path"], output_path,
        allowed_labels=labels, probabilities=probabilities, class_order=result["class_order"],
        publication_state=MODEL_SUCCESS, score=float(result["score"]),
    )


def run_ensemble(task_dir, task_spec, max_members=8):
    """Смесь лучших завершенных экспериментов: разные наборы признаков ошибаются по-разному.

    Фолды у всех экспериментов одинаковые, поэтому OOF смешиваются честно; веса подбираются
    с кросс-фиттингом, а сабмит заменяется по тому же правилу, что и после эксперимента.
    """
    from training import blend_results, compare_results, optimize_blend_weights

    task_dir = Path(task_dir)
    history = experiment_history(task_dir)
    candidates = [h for h in history if h.get("preds_file") and Path(h["preds_file"]).exists()
                  and h["recipe"] != "ensemble"]
    if len(candidates) < 2:
        raise ValueError("Для ансамбля нужно хотя бы два завершенных эксперимента.")
    greater = candidates[0]["greater_is_better"]
    candidates = sorted(candidates, key=lambda h: h["score"], reverse=greater)[:max_members]

    data = load_data(task_dir, task_spec)
    is_regression = task_spec["task_type"] == "regression"
    results = []
    for h in candidates:
        with np.load(h["preds_file"]) as archive:
            stored = dict(archive)
        order = np.asarray(h["class_order"], dtype=object)
        labels = (stored["test"].astype(float) if is_regression
                  else order[stored.get("predicted_class", stored["test"].argmax(1))])
        results.append({
            "oof": stored["oof"].astype(float), "fold_ids": stored["fold_ids"],
            "meta_known_from": stored.get("meta_known_from"),
            "predictions": labels,
            "probabilities": None if is_regression else stored["test"].astype(float),
            "class_order": h["class_order"], "score": h["score"], "score_std": h["score_std"],
            "fold_scores": h["fold_scores"], "greater_is_better": greater, "fit_time": h["fit_time"],
            "config": {"task_type": task_spec["task_type"], "split": h.get("split"), "fold_windows": h.get("fold_windows", []),
                       "fold_baselines": h.get("fold_baselines", []),
                       "n_folds": h["n_folds_done"],
                       "n_folds_done": h["n_folds_done"], "seed": 42, "fold_seed": 42},
        })
    y = data["y_train"].reset_index(drop=True)
    metric = task_spec["metric"]
    result = optimize_blend_weights(results, y, metric)
    incumbent = next((h for h in reversed(history) if h["saved"]), None)
    if incumbent and incumbent.get("fold_baselines") and not _beats_fold_baselines(
            incumbent["fold_scores"], incumbent["fold_baselines"], greater):
        # Equal-weight pairs offer complementary rankings with no fitted weights.
        # Compare them only on the same temporal folds and require majority improvement.
        incumbent_stats = {k: incumbent[k] for k in ("fold_scores", "greater_is_better")}
        strongest = None
        fixed_pairs = ((pair, blend_results([results[i] for i in pair], y, metric))
                       for pair in combinations(range(len(results)), 2))
        for pair, candidate in chain([(None, result)], fixed_pairs):
            if not _beats_fold_baselines(candidate["fold_scores"], incumbent["fold_baselines"], greater):
                continue
            comparison = compare_results(incumbent_stats, {"fold_scores": candidate["fold_scores"]})
            if comparison["n_better"] > len(candidate["fold_scores"]) / 2 and comparison["diff_mean"] > 0:
                rank = candidate["score"] * (1 if greater else -1)
                if strongest is None or rank > strongest[0]:
                    strongest = rank, pair, candidate
        if strongest is not None:
            _, pair, result = strongest
            if pair is not None:
                candidates = [candidates[i] for i in pair]
                result["config"]["blend_type"] = "equal_pair_beats_all_baselines"
    if result.get("predictions") is None and result.get("probabilities") is not None:
        # Лучшей оказалась одиночная модель из файла: метки восстанавливаем из вероятностей
        order = np.asarray(result["class_order"], dtype=object)
        result = result | {"predictions": order[np.asarray(result["probabilities"]).argmax(1)]}
    result.setdefault("config", {})
    result["config"] = result["config"] | {"n_folds": candidates[0]["n_folds_done"]}
    report = {"task_spec": task_spec, "recipe": "ensemble", "split": candidates[0].get("split"), "feature_code": None,
              "members": {f"exp{history.index(h)}": {"score": h["score"], "recipe": h["recipe"]} for h in candidates},
              "config": {"blend_weights": result["config"].get("blend_weights"),
                         "blend_type": result["config"].get("blend_type")}}
    return _record(task_dir, task_spec, data, history, result, report)


def template_id(sample):
    unique = [c for c in sample if sample[c].is_unique and sample[c].str.strip().ne("").all()]
    named = [c for c in unique if c.lower() == "id" or c.lower().endswith(("_id", "_key", "_uuid"))]
    return (named or unique or [sample.columns[0]])[0]


def verify_template_output(sample_path, task_text, output):
    """Verify the bytes that remain after the worker exits, including emergency publications."""
    from data_tools import PROBA_METRICS, REGRESSION_METRICS, detect_metric
    sample = pd.read_csv(sample_path, dtype=str, keep_default_na=False)
    raw = pd.read_csv(output, dtype=str, keep_default_na=False)
    if not raw.columns.equals(sample.columns) or len(raw) != len(sample):
        raise ValueError("Published CSV has different columns or length from template")
    ordered = [c for c in sample if sample[c].is_unique and sample[c].str.strip().ne("").all()
               and sample[c].equals(raw[c])]
    if not ordered:
        raise ValueError("Published CSV does not contain the ordered template IDs")
    key = template_id(sample[ordered])
    answers = raw.columns.drop(key)
    metric = detect_metric(task_text)
    proba = len(answers) > 1 or metric in PROBA_METRICS
    if proba or metric in REGRESSION_METRICS:
        raw[answers] = raw[answers].apply(pd.to_numeric, errors="raise")
    problems = validate_submission(raw, sample[key], list(sample), probabilities=proba)
    if metric in REGRESSION_METRICS and re.search(r"non[- ]?negative|неотрицательн", task_text, re.I):
        if (raw[answers] < 0).any().any():
            problems.append("negative regression prediction")
    if raw[answers].eq("").any().any() or problems:
        raise ValueError(f"Invalid final predictions: {problems}")
    if raw[answers].nunique().max() == 1:
        print("Warning: constant submission", file=sys.stderr, flush=True)


def schema_fallback(sample_path, task, output):
    from data_tools import detect_metric, PROBA_METRICS
    sample = pd.read_csv(sample_path, dtype=str, keep_default_na=False)
    key = template_id(sample)
    answer = [c for c in sample if c != key]
    probabilities = len(answer) > 1 or detect_metric(task) in PROBA_METRICS
    if len(answer) > 1:
        values = np.full((len(sample), len(answer)), 1 / len(answer))
    elif probabilities:
        values = np.full(len(sample), 0.5)
    else:
        mode = sample[answer[0]].mode()
        label_metric = detect_metric(task) in ("f1", "macro_f1", "accuracy", "balanced_accuracy", "weighted_f1")
        values = np.full(len(sample), mode.iloc[0] if label_metric and len(mode) else 1.)
    problems = make_submission(values, sample[key], sample_path, output, probabilities=probabilities,
                               publication_state=SCHEMA_FALLBACK)
    if problems:
        raise ValueError(problems)
    return key


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
        value = float(y.median() if task_spec["metric"] == "mae" or _nonnegative(task_spec) else y.mean())
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
    if frame.drop(columns=identifier).nunique().max() == 1:
        print("Warning: constant submission", file=sys.stderr, flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    args = json.load(sys.stdin.buffer)
    output_directory.set(args.pop("output_dir", None))
    receipt_directory.set(args.pop("receipt_dir", None))
    if args["recipe"] == "ensemble":
        report = run_ensemble(Path.cwd(), args["task_spec"])
    else:
        report = run_experiment(Path.cwd(), **args)
    print(json.dumps(report, ensure_ascii=False, default=str), flush=True)
