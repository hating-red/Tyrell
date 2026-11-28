import hashlib
import json
import math
import re
import os
import sys
import shutil
import tempfile
import time
from contextvars import ContextVar

from pathlib import Path

from time_budget import available_cpus, limit_openmp

limit_openmp()

import numpy as np
import pandas as pd

from data_tools import (INTERVAL_FEATURE_SUFFIXES, REFERENCE_DATE_SUFFIX, detect_split, load_data,
                        parse_event_datetime, label_available_at)
from publication import MODEL_SUCCESS, TRAINED_FALLBACK, SCHEMA_FALLBACK, csv_digest, prepare_publication, receipt_directory
from submission import make_submission, target_filler, validate_submission

PREDS_DIR = ".tyrell_preds"       # OOF и предсказания test завершенных экспериментов для ансамбля
output_directory = ContextVar("output_directory", default=None)

# Ключи конфига, которые имеют смысл для всего рецепта auto, а не для одной модели
AUTO_KEYS = ("features", "cat_encoding", "target_encoding", "refit_full", "entity_history", "log_target",
             "ad_history", "ad_history_window", "ad_history_smoothing", "event_activity", "causal_history",
             "history_window", "history_smoothing", "mae_shift", "recency_days", "mae_objective",
             "auto_features", "rare_min_freq", "catboost_ordered_ctr", "es_metric", "event_calendar",
             "fold_preprocess", "sqrt_target", "history_since_positive", "regularized_ap")
_HOURLY_SUFFIXES = ("__mean24h", "__mean168h", "__delta1_3h", "__dev7d")


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


def prepare_features(task_dir, task_spec, config, *, data=None, feature_code=None):
    """Prepare identical inputs for outer training and hyperparameter selection."""
    if data is None:
        data = load_data(task_dir, task_spec)
    from leak_audit import LeakAuditError

    audited = task_spec.get("audit_required", False)
    if audited and feature_code:
        raise LeakAuditError("custom transforms are not certified for audited folds")
    train, test = data["X_train"], data["X_test"]
    reference_dates = [name for name in train if name.endswith(REFERENCE_DATE_SUFFIX)]
    split_train = train.drop(columns=reference_dates) if reference_dates else train
    split_test = test.drop(columns=reference_dates) if reference_dates else test
    split = detect_split(split_train, split_test, time_column=task_spec.get("time_column"),
                         group_column=task_spec.get("group_column"))
    temporal = (split or {}).get("type") == "time"
    X_train, X_test = train.reset_index(drop=True), test.reset_index(drop=True)
    if temporal:
        clock = split["column"]
        if feature_code:
            raise ValueError("Temporal features must use the row-local transformation")
        if not pd.api.types.is_numeric_dtype(X_train[clock]):
            for frame in (X_train, X_test):
                frame[clock] = parse_event_datetime(frame[clock])
    if feature_code:
        X_train, X_test = apply_feature_code(feature_code, X_train, X_test)
    cfg = dict(config or {})
    if audited and temporal and any(cfg.get(key) for key in ("target_encoding", "entity_history", "ad_history")):
        raise LeakAuditError("temporal configuration uses an uncertified legacy feature group")
    hourly_extended = bool(cfg.pop("hourly_extended", False))
    interval_features = bool(cfg.pop("interval_features", False))
    dropped = [name for name in X_train if
               (audited and not hourly_extended and name.endswith(_HOURLY_SUFFIXES)) or
               (not interval_features and name.endswith(INTERVAL_FEATURE_SUFFIXES)) or
               (name in reference_dates and not (temporal and cfg.get("fold_preprocess")))]
    if dropped:
        X_train, X_test = X_train.drop(columns=dropped), X_test.drop(columns=dropped)
    auto_features = bool(cfg.pop("auto_features", False))
    if auto_features and temporal:
        from features import add_auto_features
        transformed = add_auto_features(X_train, X_test, row_local=True)
        if audited:
            from leak_audit import AuditBudget, check_equal, sample_positions

            def verify_calendar(counts, check_time):
                a = sample_positions(len(X_train), limit=counts[0])
                b = sample_positions(len(X_test), limit=counts[1])
                replayed = add_auto_features(X_train.iloc[a], X_test.iloc[b], row_local=True)
                check_equal(transformed[0].iloc[a], replayed[0])
                check_equal(transformed[1].iloc[b], replayed[1])
                return dict(features=[c for c in transformed[0] if c not in X_train], rule="current event values only")

            AuditBudget().sampled("calendar", [len(X_train), len(X_test)], verify_calendar)
        X_train, X_test = transformed
    elif auto_features:
        cfg["auto_features"] = True

    for key in ("n_folds", "fold_seed", "split", "n_jobs"):
        cfg.pop(key, None)
    fixed = {"n_folds": 5, "fold_seed": 42, "n_jobs": available_cpus(), "split": split, "audit_required": audited}
    for key in ("label_delay_unit", "positive_class"):
        if key in task_spec:
            fixed[key] = task_spec[key]
    return X_train, X_test, cfg | fixed


def run_experiment(task_dir, task_spec, config=None, time_budget=300, recipe="single", feature_code=None):
    from training import METRICS, auto_train, compare_results, train_and_predict

    task_dir = Path(task_dir)
    data = load_data(task_dir, task_spec)
    task_type, metric = task_spec["task_type"], task_spec["metric"]
    if metric not in METRICS or task_type not in METRICS[metric].task_types:
        raise ValueError("Неизвестная метрика или метрика не соответствует типу задачи.")
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

    from leak_audit import LeakAuditError

    config = dict(config or {})
    auto_features = bool(config.get("auto_features"))
    hourly_extended = bool(config.get("hourly_extended"))
    interval_features = bool(config.get("interval_features"))
    try:
        X_train, X_test, cfg = prepare_features(task_dir, task_spec, config, data=data, feature_code=feature_code)
    except LeakAuditError as exc:
        return {"skipped": f"leak_audit_failed: {exc}"}
    y_train = data["y_train"].reset_index(drop=True)
    fixed = {key: cfg.pop(key) for key in ("n_folds", "fold_seed", "n_jobs", "audit_required", "split")}
    for key in ("label_delay_unit", "positive_class"):
        if key in cfg:
            fixed[key] = cfg.pop(key)
    split = fixed["split"]

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

    max_seeds = int(cfg.pop("max_seeds", None) or 3)
    cfg["params"] = dict(cfg.get("params") or {})
    for key in ("num_threads", "nthread", "thread_count", "n_jobs"):
        if key in cfg["params"]:
            cfg["params"][key] = fixed["n_jobs"]
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
        "config": {k: v for k, v in cfg.items() if k != "features"} | {
            "auto_features": auto_features, "hourly_extended": hourly_extended, "interval_features": interval_features},
    }
    if config.get("features") is not None:
        report["config"]["features"] = config["features"]
    report["feature_candidates"] = result.get("feature_candidates", {}) | {
        "hourly_extended": any(name.endswith(_HOURLY_SUFFIXES) for name in data["X_train"]),
        "interval_features": any(name.endswith(INTERVAL_FEATURE_SUFFIXES) for name in data["X_train"]),
    }
    if "auto_stop" in result:
        report["auto_stop"] = result["auto_stop"]
    if cfg.get("profile_categories"):
        report["category_probes"] = result.get("category_probes")
    for key in ("model_inputs", "fold_diagnostics"):
        if key in result:
            report[key] = result[key]
    return _record(task_dir, task_spec, data, history, result, report)


def run_prefix_tuning(task_dir, task_spec, config, time_budget):
    from training import tune_backend

    started = time.monotonic()
    data = load_data(task_dir, task_spec)
    X_train, X_test, prepared = prepare_features(task_dir, task_spec | {"audit_required": True}, config, data=data)
    prepared.pop("max_seeds", None)
    stats = {}
    left = max(0., time_budget - (time.monotonic() - started))
    tuned = tune_backend(prepared["model"], X_train, data["y_train"], X_test,
                         task_spec["task_type"], task_spec["metric"], prepared, left, diagnostics=stats)
    return {"tuned_params": tuned, "tune_diagnostics": stats}


def _beats_fold_baselines(scores, baselines, greater):
    if not baselines or len(scores) != len(baselines):
        return False
    return all((score > base["score"] if greater else score < base["score"])
               for score, base in zip(scores, baselines))


def _passes_quality_gate(control, candidate, compare_results):
    if control.get("fold_signature") != candidate.get("fold_signature"):
        return False
    paired = compare_results(control, candidate)
    folds = len(paired["diffs"])
    direction = 1 if candidate["greater_is_better"] else -1

    def baseline_wins(report):
        return sum(direction * (score - base["score"]) > 0
                   for score, base in zip(report["fold_scores"], report.get("fold_baselines", [])))

    return (paired["n_better"] > folds / 2 and paired["diff_mean"] > paired["diff_std"] / math.sqrt(folds)
            and baseline_wins(candidate) >= baseline_wins(control))


def _archive_metadata(result):
    cfg = result["config"]
    return {"effective_model": cfg.get("model"), "effective_params": cfg.get("params"),
            "effective_seed": cfg.get("seed"), "effective_class_weight": cfg.get("class_weight"),
            "best_iterations": result.get("best_iterations", []), "fold_baselines": cfg.get("fold_baselines", []),
            "fold_windows": cfg.get("fold_windows", []), "early_stopping_windows": cfg.get("early_stopping_windows", []),
            "target_bounds": result.get("target_bounds"), "repair_filler": result.get("repair_filler", cfg.get("repair_filler")),
            "fold_target_bounds": result.get("fold_target_bounds"), "integer_target": result.get("integer_target"),
            "fold_integer_target": result.get("fold_integer_target"), "postprocess": result.get("postprocess")}


def _archive_oof(task_dir, result, stem):
    archive = Path(os.environ.get("TYRELL_OOF_RUN_DIR", task_dir))
    test = result["probabilities"] if result["probabilities"] is not None else result["predictions"]
    arrays = {"oof": np.asarray(result["oof"], dtype=np.float32),
              "test": np.asarray(test, dtype=np.float64), "fold_ids": result["fold_ids"]}
    if result.get("meta_known_from") is not None:
        arrays["meta_known_from"] = result["meta_known_from"]
    if result["class_order"] is not None:
        arrays["predicted_class"] = pd.Index(result["class_order"]).get_indexer(result["predictions"])
    for root in dict.fromkeys((archive, task_dir)):
        try:
            preds_dir = root / PREDS_DIR
            preds_dir.mkdir(exist_ok=True)
            path = preds_dir / f"{stem}.npz"
            np.savez_compressed(path, **arrays)
            return str(path)
        except OSError as error:
            if root == task_dir:
                raise
            print(f"OOF_ARCHIVE unavailable: {error}", file=sys.stderr, flush=True)


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
    quality_pass = None
    quality_audit = None
    if previous is not None:
        same_folds = signature == previous.get("fold_signature")
        if same_folds:
            # Не берем «везучую» модель: новый результат должен быть лучше на большинстве
            # фолдов и с приростом больше шума, а не просто чуть выше в среднем
            comparison = compare_results(
                {"fold_scores": previous["fold_scores"], "greater_is_better": previous["greater_is_better"]},
                {"fold_scores": result["fold_scores"], "greater_is_better": result["greater_is_better"]},
            )
        candidate = {"fold_signature": signature, "fold_scores": result["fold_scores"],
                     "greater_is_better": greater, "fold_baselines": baselines}
        if task_spec.get("audit_required") or report.get("recipe") == "ensemble":
            better = complete and (not previous["complete"] or
                                   _passes_quality_gate(previous, candidate, compare_results))
        else:
            better = (complete and not previous["complete"]) or (same_folds and comparison["accept"])
            if same_folds and baselines and previous.get("fold_baselines"):
                old_wins = _beats_fold_baselines(previous["fold_scores"], previous["fold_baselines"], greater)
                new_wins = _beats_fold_baselines(result["fold_scores"], baselines, greater)
                if old_wins != new_wins:
                    better = new_wins and complete
            if report.get("config", {}).get("ad_history"):
                better = bool(better and same_folds and comparison["n_better"] > n_folds / 2)
        if (report.get("recipe") == "ensemble" and complete and same_folds
                and (report.get("refit") or {}).get("same_incumbent")
                and np.array_equal(previous["fold_scores"], result["fold_scores"])):
            better = True
        if task_spec.get("audit_required") and report.get("recipe") != "ensemble":
            from training import DEFAULT_CONFIG

            quality_flags = ["rare_min_freq", "catboost_ordered_ctr", "es_metric", "event_calendar",
                             "fold_preprocess", "sqrt_target", "regularized_ap", "log_target",
                             "mae_objective", "history_smoothing", "history_since_positive"]
            if (report.get("split") or {}).get("type") != "time":
                quality_flags.append("auto_features")
            config = report.get("config", {})
            defaults = ({key: DEFAULT_CONFIG[key] for key in quality_flags} | {"interval_features": False})
            trial_flags = [key for key, default in defaults.items() if config.get(key, default) not in
                           (default, previous.get("config", {}).get(key, default))]
            if trial_flags:
                control_config = defaults | config | {key: defaults[key] for key in trial_flags}
                control = next((row for row in reversed(history)
                                if row.get("complete") and row.get("recipe") == report.get("recipe")
                                and (defaults | row.get("config", {})) == control_config), None)
                candidate = {"fold_signature": signature, "fold_scores": result["fold_scores"],
                             "greater_is_better": greater, "fold_baselines": baselines}
                quality_pass = bool(complete and control and _passes_quality_gate(control, candidate, compare_results))
                better = quality_pass and (not previous["complete"] or
                                           _passes_quality_gate(previous, candidate, compare_results))
                gains = (compare_results(control, candidate)["diffs"] if control and
                         signature == control.get("fold_signature") else None)
                quality_audit = {"flags": {key: config[key] for key in trial_flags},
                                 "paired_fold_gains": gains, "accepted": quality_pass, "reason": ("passed" if quality_pass else
                                     "missing_paired_control" if control is None else "paired_gate")}
                print("QUALITY_FLAGS " + json.dumps(quality_audit), file=sys.stderr, flush=True)
    preds_file = (_archive_oof(task_dir, result, str(len(history)))
                  if complete and "fold_ids" in result else None)
    if result.get("member_results"):
        members = []
        for name, candidate in result["member_results"].items():
            if "fold_ids" not in candidate or candidate["config"]["n_folds_done"] != candidate["config"]["n_folds"]:
                continue
            stored = dict(report)
            stored.pop("member_candidates", None)
            stored["recipe"] = "member"
            stored["config"] = report["config"] | {
                key: candidate["config"][key] for key in (
                    "model", "params", "seed", "class_weight", "log_target", "sqrt_target", "mae_objective",
                    "regularized_ap", "es_metric") if key in candidate["config"]
            }
            stored.update(_archive_metadata(candidate), score=candidate["score"], score_std=candidate["score_std"],
                          fold_scores=candidate["fold_scores"], greater_is_better=greater, complete=True,
                          class_order=candidate["class_order"], saved=False, quality_flags_pass=quality_pass,
                          fit_time=candidate["fit_time"], candidate_id=f"{len(history)}.{name}",
                          n_folds_done=candidate["config"]["n_folds_done"],
                          fold_signature=hashlib.sha256(np.asarray(candidate["fold_ids"], dtype="<i4").tobytes()).hexdigest(),
                          preds_file=_archive_oof(task_dir, candidate, f"{len(history)}-{name}"))
            members.append(stored)
        if members:
            report["member_candidates"] = members

    output = _output_path(task_dir)
    staged_csv = task_dir / ".tyrell_candidate.csv"
    problems = _write_submission(staged_csv, task_spec, data, result) if better else []
    report = report | {
        "score": result["score"], "score_std": result["score_std"], "metric": task_spec["metric"],
        "fold_scores": [float(s) for s in result.get("fold_scores", [])],
        "greater_is_better": result["greater_is_better"], "complete": complete, "n_folds_done": n_folds_done,
        **_archive_metadata(result),
        "saved": better and not problems, "problems": problems,
        "best_saved_score": result["score"] if better and not problems else (previous["score"] if previous else None),
        "vs_saved": comparison and {k: comparison[k] for k in ("accept", "diff_mean", "diff_std", "n_better", "diffs")},
        "quality_flags_pass": quality_pass,
        "quality_flags_audit": quality_audit,
        "fold_signature": signature,
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
    if "TYRELL_OOF_RUN_DIR" in os.environ:
        archive_history = Path(os.environ["TYRELL_OOF_RUN_DIR"]) / "experiments.json"
        staged_archive = archive_history.with_suffix(".tmp")
        try:
            shutil.copyfile(history_file, staged_archive)
            staged_archive.replace(archive_history)
        except OSError as error:
            print(f"OOF_ARCHIVE unavailable: {error}", file=sys.stderr, flush=True)
    if task_spec.get("audit_required") and report["saved"] and comparison:
        rules = {"auto_features": ("fit-only statistics" if (report.get("split") or {}).get("type") != "time"
                                   else "current event values only"),
                 "event_activity": "only registrations strictly before the event",
                 "causal_history": "only fit-fold labels available strictly before the event",
                 "hourly_extended": "interval end and recorded publication both strictly before the event"}
        groups = {key: value for key, value in rules.items() if report.get("config", {}).get(key)}
        if groups:
            print("FEATURE_ACCEPTED " + json.dumps({"groups": groups, "per_fold_gain": comparison["diffs"]}),
                  file=sys.stderr, flush=True)
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
    regression = task_type == "regression"
    probabilities = not regression and task_spec.get("probabilities", False)
    predictions = result["predictions"]
    if regression and _nonnegative(task_spec) and np.asarray(predictions).dtype.kind in "iuf":
        predictions = np.where(np.isfinite(predictions), np.maximum(predictions, 0), predictions)
    sample = pd.read_csv(data["sample_path"], nrows=0)
    answer_count = len(sample.columns) - 1
    filler = result.get("repair_filler", result.get("config", {}).get("repair_filler"))
    if regression and answer_count != 1:
        return ["Неподдерживаемый формат: мультитаргетная регрессия"]
    if not regression and (probabilities or answer_count > 1):
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
            if filler is not None:
                filler = filler[classes.index(positive)]
    labels = set(result["class_order"]) if not regression and not probabilities else None
    if labels is not None and filler is not None:
        filler = result["class_order"][int(np.argmax(filler))]
    if regression and "y_train" in data:
        y = _fallback_labels(data, task_spec)
        if len(y) and np.asarray(predictions).dtype.kind in "iuf":
            center = float(y.median())
            q1, q3 = y.quantile([.25, .75])
            finite = np.asarray(predictions)[np.isfinite(predictions)]
            scale = max(q3 - q1, abs(center) * .1, 1e-8)
            if len(finite) and abs(float(np.median(finite)) - center) > 3 * scale:
                print("Warning: PREDICTION_DRIFT regression median differs from train",
                      file=sys.stderr, flush=True)
    if probabilities and task_spec["metric"] == "logloss" and filler is not None:
        values = np.asarray(predictions)
        if values.dtype.kind in "iuf" and np.isfinite(values).all() and (
                np.abs(values.mean(axis=0) - np.asarray(filler)) > .25).any():
            print("Warning: PREDICTION_DRIFT mean probability differs from train prior",
                  file=sys.stderr, flush=True)
    return make_submission(
        predictions, data["test_ids"], data["sample_path"], output_path,
        allowed_labels=labels, probabilities=probabilities, class_order=result["class_order"],
        answer_kind="regression" if regression else "probabilities" if probabilities else "labels", filler=filler,
        publication_state=MODEL_SUCCESS, score=float(result["score"]),
    )


def _ensemble_target_info(data, task_spec, split, fold_ids):
    """Reconstruct fit bounds for archives written before the target metadata existed."""
    y = pd.to_numeric(data["y_train"], errors="coerce").to_numpy(dtype=float)
    observed = np.isfinite(y)
    folds = np.unique(fold_ids[fold_ids >= 0])
    temporal = (split or {}).get("type") == "time"
    if temporal:
        from training import _time_values
        column = split["column"]
        times = _time_values(data["X_train"][column])
        releases = (label_available_at(times, y, task_spec["label_delay_unit"])
                    if task_spec.get("label_delay_unit") else times)
        cutoff = _time_values(data["X_test"][column]).min()
        masks = [observed & releases.lt(times.iloc[np.flatnonzero(fold_ids == k)].min()).to_numpy() for k in folds]
        observed &= releases.lt(cutoff).to_numpy()
    else:
        masks = [observed & (fold_ids != k) for k in folds]

    def bounds(mask):
        values = y[mask]
        return (float(values.min()), float(values.max())) if len(values) else None

    def integer(mask):
        values = y[mask]
        return bool(len(values) and np.equal(values, np.round(values)).all())

    return {"target_bounds": bounds(observed), "fold_target_bounds": [bounds(mask) for mask in masks],
            "integer_target": integer(observed), "fold_integer_target": [integer(mask) for mask in masks]}


def _refit_ensemble(task_dir, task_spec, data, selected, deadline):
    from features import add_key_columns
    from leak_audit import LeakAuditError
    from training import DEFAULT_CONFIG, train_and_predict

    predictions, details = {}, []
    setups, estimates = {}, {}
    y = data["y_train"].reset_index(drop=True)
    for rank, (_, member) in enumerate(selected):
        model = member.get("effective_model") or member["config"].get("model")
        if not model or not member.get("best_iterations"):
            continue
        iterations = member["best_iterations"]
        rounds = max(1, round(np.median(iterations) if model == "linear" else np.mean(iterations) * 1.1))
        estimates[rank] = max(2., member["fit_time"] / max(2, len(iterations)))
        setups[rank] = (member, model, rounds, int(member.get("effective_seed") or member["config"].get("seed", 42)))
    for offset in (0, 1, 2):
        for rank, (member, model, rounds, original_seed) in setups.items():
            if offset and (rank >= 3 or rank not in predictions):
                continue
            usable = deadline - time.monotonic() - 10.
            estimate = estimates[rank]
            if usable < 1.35 * estimate:
                continue
            began = time.monotonic()
            try:
                original = member["config"]
                X, X_test, config = prepare_features(task_dir, task_spec, original, data=data,
                                                     feature_code=member.get("feature_code"))
                config.pop("target_encoding", None)
                if member.get("te_keys"):
                    X, X_test, config["target_encoding"] = add_key_columns(
                        X, X_test, member["te_keys"], raw_frames=(data["X_train"], data["X_test"]))
                cfg = {key: val for key, val in config.items() if key in DEFAULT_CONFIG}
                cfg.update(model=model, seed=original_seed + offset,
                           params=member.get("effective_params") or original.get("params") or {},
                           class_weight=member.get("effective_class_weight", original.get("class_weight")))
                result = train_and_predict(X, y, X_test, task_spec["task_type"], task_spec["metric"], cfg,
                                           time_budget=min(usable, 2.5 * estimate), refit_rounds=rounds)
                if result["stopped"]:
                    continue
                predictions.setdefault(rank, []).append(result["test_predictions"])
                details.append({"experiment": selected[rank][0], "model": model, "seed": result["seed"],
                                "rounds": rounds, "seconds": round(time.monotonic() - began, 3)})
                estimates[rank] = max(2., time.monotonic() - began)
            except LeakAuditError:
                raise
            except (OSError, ValueError, RuntimeError, TimeoutError) as error:
                print(f"ENSEMBLE_REFIT skipped model={model} seed={original_seed + offset} error={error}",
                      file=sys.stderr, flush=True)
    return {rank: np.mean(values, axis=0) for rank, values in predictions.items()}, details


def run_ensemble(task_dir, task_spec, max_members=16, *, time_budget=None, refit=False):
    """Cross-fit the selection and meta model, then optionally refit the final members."""
    from leak_audit import audit_log
    from training import _constrain_ensemble, apply_ensemble_meta, compare_results, select_ensemble

    started = time.monotonic()
    deadline = started + time_budget if time_budget is not None else float("inf")
    task_dir = Path(task_dir)
    history = experiment_history(task_dir)
    incumbent_pair = next(((i, row) for i, row in reversed(list(enumerate(history))) if row["saved"]), None)
    incumbent_id, incumbent = incumbent_pair or (None, None)
    roster = [(item.get("candidate_id", i), item) for i, row in enumerate(history)
              for item in [row, *row.get("member_candidates", [])]]
    eligible = [(i, row) for i, row in roster
                if row.get("preds_file") and Path(row["preds_file"]).exists() and row["recipe"] != "ensemble"
                and row.get("quality_flags_pass") is not False
                and (not incumbent or not incumbent.get("complete") or not incumbent.get("fold_signature")
                     or row.get("fold_signature") == incumbent["fold_signature"])]
    if not eligible or (len(eligible) < 2 and not refit):
        raise ValueError("Insufficient eligible completed experiments for ensemble")
    data = load_data(task_dir, task_spec)
    y = data["y_train"].reset_index(drop=True)
    is_regression = task_spec["task_type"] == "regression"
    width = 1 if is_regression else max(2, y.nunique())
    capacity = max(2, min(96, 450_000_000 // max(1, (len(y) + len(data.get("X_test", ()))) * width * 8)))
    # Membership of the memory-bounded roster depends only on chronology of the journal.
    if len(eligible) > capacity:
        eligible = [eligible[0], *eligible[-capacity + 1:]]
    results = []
    for _, row in eligible:
        with np.load(row["preds_file"]) as archive:
            stored = dict(archive)
        if (task_spec.get("label_delay_unit") and (row.get("split") or {}).get("type") == "time"
                and "meta_known_from" not in stored):
            column = row["split"]["column"]
            times = parse_event_datetime(data["X_train"][column])
            available = label_available_at(times, y.to_numpy(), task_spec["label_delay_unit"])
            folds = stored["fold_ids"]
            cutoffs = pd.Index([times.iloc[np.flatnonzero(folds == k)].min() for k in np.unique(folds[folds >= 0])])
            stored["meta_known_from"] = cutoffs.searchsorted(available, side="right")
        order = np.asarray(row["class_order"], dtype=object)
        values = stored["test"].astype(float)
        labels = values if is_regression else order[stored.get("predicted_class", values.argmax(1))]
        params = row["config"] | {
            "task_type": task_spec["task_type"], "split": row.get("split"), "fold_windows": row.get("fold_windows", []),
            "fold_baselines": row.get("fold_baselines", []), "repair_filler": row.get("repair_filler"),
            "n_folds": row["n_folds_done"],
            "n_folds_done": row["n_folds_done"], "seed": 42, "fold_seed": 42,
            "model": row.get("effective_model") or row["config"].get("model") or "auto",
            "class_weight": row.get("effective_class_weight", row["config"].get("class_weight")),
        }
        if is_regression:
            metadata = {key: row.get(key) for key in
                        ("target_bounds", "fold_target_bounds", "integer_target", "fold_integer_target")}
            if metadata["target_bounds"] is None:
                metadata = (_ensemble_target_info(data, task_spec, row.get("split"), stored["fold_ids"])
                            if "X_train" in data else {})
            params.update(metadata)
        results.append({"oof": stored["oof"], "fold_ids": stored["fold_ids"], "meta_known_from": stored.get("meta_known_from"),
                        "predictions": labels, "probabilities": None if is_regression else values,
                        "class_order": row["class_order"], "score": row["score"], "score_std": row["score_std"],
                        "fold_scores": row["fold_scores"], "greater_is_better": row["greater_is_better"],
                        "fit_time": row["fit_time"], "config": params})

    selection_deadline = deadline
    if refit and time_budget is not None:
        selection_deadline = deadline - min(180., time_budget * .45)
    result, plan, stats = select_ensemble(results, y, task_spec["metric"], limit=min(16, max_members),
                                          deadline=selection_deadline, nonnegative=_nonnegative(task_spec))
    indices, predict = plan
    candidate = {"fold_signature": eligible[0][1].get("fold_signature"),
                 "fold_scores": result["fold_scores"], "greater_is_better": result["greater_is_better"],
                 "fold_baselines": result["config"].get("fold_baselines", [])}
    reuse_meta = False
    if refit and incumbent and incumbent.get("complete") and not _passes_quality_gate(incumbent, candidate, compare_results):
        positions = {tag: i for i, (tag, _) in enumerate(eligible)}
        cfg = incumbent.get("config", {})
        if incumbent_id in positions:
            i = positions[incumbent_id]
            indices = [i]
            result = results[i] | {"config": results[i]["config"] | {"blend_type": "best_single", "blend_space": "prob"}}
            bounds = result["config"].get("target_bounds")
            predict = lambda p: _constrain_ensemble(p[0], bounds, not is_regression, _nonnegative(task_spec),
                                                    result["config"].get("repair_filler"))
        elif (cfg.get("final_meta") and cfg.get("selected") and all(tag in positions for tag in cfg["selected"])
              and incumbent.get("preds_file") and Path(incumbent["preds_file"]).is_file()):
            indices = [positions[tag] for tag in cfg["selected"]]
            reference = results[indices[0]]
            with np.load(incumbent["preds_file"]) as archive:
                stored = dict(archive)
            values = stored["test"]
            labels = values if is_regression else np.asarray(reference["class_order"], dtype=object)[
                stored.get("predicted_class", values.argmax(1))]
            result = reference | {"predictions": labels, "probabilities": None if is_regression else values,
                                  "oof": stored["oof"], "score": incumbent["score"], "score_std": incumbent["score_std"],
                                  "fold_scores": incumbent["fold_scores"], "postprocess": incumbent.get("postprocess"),
                                  "config": reference["config"] | cfg}
            bounds = reference["config"].get("target_bounds")
            predict = lambda p: apply_ensemble_meta(cfg["final_meta"], p, task_spec["task_type"], task_spec["metric"],
                                                    bounds, _nonnegative(task_spec))
            reuse_meta = True
    selected = [eligible[i] for i in indices]
    selected_ids = [idx for idx, _ in selected]
    report_config = {key: result["config"].get(key) for key in
                     ("blend_type", "blend_space", "blend_weights", "fold_meta", "final_meta")}
    report_config["selected"] = selected_ids
    if not reuse_meta:
        for key in ("fold_meta", "final_meta"):
            meta = report_config.get(key)
            for record in (meta if isinstance(meta, list) else [meta] if meta else []):
                record["members"] = [eligible[i][0] for i in record["members"]]
    stats["pool"] = [eligible[i][0] for i in stats["pool"]]
    stats["selected"] = selected_ids
    stats["strategy"] = report_config["blend_type"]
    stats["fold_scores"] = result["fold_scores"]
    stats["best_single"] = eligible[stats["best_single"]][0]
    stats["duplicates"] = [[eligible[i][0], eligible[j][0], value] for i, j, value in stats["duplicates"]]
    report = {"task_spec": task_spec, "recipe": "ensemble", "split": eligible[0][1].get("split"), "feature_code": None,
              "members": {f"exp{i}": {"score": row["score"], "recipe": row["recipe"],
                                     "model": row.get("effective_model") or row["config"].get("model"),
                                     "variant": row["config"]} for i, row in eligible if i in stats["pool"]},
              "config": report_config, "ensemble": stats}

    if task_spec.get("audit_required"):
        temporal = (report["split"] or {}).get("type") == "time"
        audit_log("pass", "ensemble_meta", sample_sizes=[m["fit_rows"] for m in report_config.get("fold_meta") or []],
                  rule="only earlier folds and released labels" if temporal else "only other validation folds")
    if refit and deadline - time.monotonic() > 12.:
        began = time.monotonic()
        fresh, fits = _refit_ensemble(task_dir, task_spec, data, selected, deadline)
        if fresh:
            key = "predictions" if is_regression else "probabilities"
            values = np.stack([fresh.get(k, results[i][key]) for k, i in enumerate(indices)])
            predictions = predict(values)
            if is_regression:
                result["predictions"] = predictions
            else:
                result["probabilities"] = predictions
                weights = np.ones(predictions.shape[1])
                if report_config["blend_type"] == "best_single" and not result.get("postprocess"):
                    post = selected[0][1].get("postprocess") or {}
                    stored_weights = post.get("weights", {})
                    if isinstance(stored_weights, dict):
                        weights = [stored_weights.get(str(c), stored_weights.get(c, 1.)) for c in result["class_order"]]
                result["predictions"] = np.asarray(result["class_order"], dtype=object)[(predictions * weights).argmax(1)]
            same = ((report_config["blend_type"] == "best_single" and selected_ids == [incumbent_id])
                    or bool(incumbent and all(incumbent.get("config", {}).get(key) == report_config[key]
                                              for key in ("blend_type", "blend_space", "selected"))))
            report["refit"] = {"seconds": round(time.monotonic() - began, 3), "fits": fits,
                               "same_incumbent": same}
    print("ENSEMBLE " + json.dumps(stats, ensure_ascii=False), file=sys.stderr, flush=True)
    result["fit_time"] = time.monotonic() - started
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
    regression = metric in REGRESSION_METRICS
    proba = not regression and (len(answers) > 1 or metric in PROBA_METRICS)
    if proba or regression:
        raw[answers] = raw[answers].apply(pd.to_numeric, errors="raise")
    kind = "regression" if regression else "probabilities" if proba else "labels"
    problems = validate_submission(raw, sample[key], list(sample), answer_kind=kind)
    if metric in REGRESSION_METRICS and re.search(r"non[- ]?negative|неотрицательн", task_text, re.I):
        if (raw[answers] < 0).any().any():
            problems.append("negative regression prediction")
    if raw[answers].eq("").any().any() or problems:
        raise ValueError(f"Invalid final predictions: {problems}")
    if raw[answers].nunique().max() == 1:
        print("Warning: constant submission", file=sys.stderr, flush=True)


def schema_fallback(sample_path, task, output):
    from data_tools import detect_metric, PROBA_METRICS, REGRESSION_METRICS
    sample = pd.read_csv(sample_path, dtype=str, keep_default_na=False)
    key = template_id(sample)
    answer = [c for c in sample if c != key]
    metric = detect_metric(task)
    regression = metric in REGRESSION_METRICS
    probabilities = not regression and (len(answer) > 1 or metric in PROBA_METRICS)
    kind = "regression" if regression else "probabilities" if probabilities else "labels"
    if regression and len(answer) > 1:
        values = np.ones((len(sample), len(answer)))
    elif len(answer) > 1:
        values = np.full((len(sample), len(answer)), 1 / len(answer))
    elif probabilities:
        values = np.full(len(sample), 0.5)
    else:
        mode = sample[answer[0]].mode()
        label_metric = detect_metric(task) in ("f1", "macro_f1", "accuracy", "balanced_accuracy", "weighted_f1")
        values = np.full(len(sample), mode.iloc[0] if label_metric and len(mode) else 1.)
    problems = make_submission(values, sample[key], sample_path, output, answer_kind=kind,
                               publication_state=SCHEMA_FALLBACK)
    if problems:
        raise ValueError(problems)
    return key


def _fallback_labels(data, task_spec, *, profile=False):
    raw = data["y_train"]
    y = raw.dropna()
    if task_spec["task_type"] == "regression":
        y = pd.to_numeric(y, errors="coerce")
        y = y[np.isfinite(y)]
    if task_spec.get("label_delay_unit") or (task_spec.get("audit_required") and "X_train" in data):
        split = detect_split(data["X_train"], data["X_test"], time_column=task_spec.get("time_column"),
                             group_column=task_spec.get("group_column"))
        if task_spec.get("label_delay_unit") and (not split or split["type"] != "time"):
            raise ValueError("A duration target needs an event clock for the early submission")
        if (split or {}).get("type") == "time":
            from training import _time_values
            col = split["column"]
            train_time = _time_values(data["X_train"].loc[y.index, col])
            test_time = _time_values(data["X_test"][col])
            unit = task_spec.get("label_delay_unit")
            available = label_available_at(train_time, y, unit) if unit else train_time
            y = y[available < test_time.min()] if test_time.notna().all() else y.iloc[:0]
            if profile:
                past = _time_values(data["X_train"][col]).lt(test_time.min())
                raw = raw.loc[past & (raw.isna() | raw.index.isin(y.index))]
    if profile:
        return y, float(raw.isna().mean()) if len(raw) else 0.
    return y


def write_fallback(task_dir, task_spec):
    """Всегда публикует прогноз текущего запуска по доступным обучающим меткам."""
    task_dir = Path(task_dir)
    data = load_data(task_dir, task_spec)
    y, missing_fraction = _fallback_labels(data, task_spec, profile=True)
    if y.empty:
        raise ValueError("No eligible training labels to calculate a fallback prediction")
    sample = pd.read_csv(data["sample_path"], nrows=0)
    n = len(data["test_ids"])
    regression = task_spec["task_type"] == "regression"
    if regression:
        if len(sample.columns) != 2:
            return ["Неподдерживаемый формат: мультитаргетная регрессия"]
        value, reason = target_filler(y.to_numpy(), task_spec["metric"])
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
                value = float(shares.get(positive, 0.0))
                predictions = np.full(n, value, dtype=float)
            else:
                answers = [c for c in sample.columns if c != data["id_column"]]
                by_label = {str(label): share for label, share in shares.items()}
                if not set(by_label).issubset(answers):
                    return ["Training classes are absent from the probability template"]
                predictions = pd.DataFrame({col: np.full(n, by_label.get(col, 0.0)) for col in answers})
                value = [float(by_label.get(col, 0.0)) for col in answers]
            labels = None
            reason = "class_prior"
        else:
            probabilities, labels = False, set(class_order)
            value, reason = shares.idxmax(), "mode"
            predictions = np.full(n, value, dtype=object)
    print(f"TRAIN_EDA available_labels={len(y)} target_missing_fraction={missing_fraction:.6f} "
          f"fallback={value!r} reason={reason}", file=sys.stderr, flush=True)
    return make_submission(
        predictions, data["test_ids"], data["sample_path"], _output_path(task_dir),
        allowed_labels=labels, probabilities=probabilities, class_order=class_order,
        answer_kind="regression" if regression else "probabilities" if probabilities else "labels",
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
    regression = task_spec["task_type"] == "regression"
    probabilities = not regression and (task_spec.get("probabilities", False) or len(sample.columns) > 2)
    labels = set(data["y_train"].dropna()) if not regression and not probabilities else None
    if labels and all(isinstance(label, str) for label in labels):
        frame = pd.read_csv(_output_path(task_dir), dtype=str, keep_default_na=False)
    kind = "regression" if regression else "probabilities" if probabilities else "labels"
    problems = validate_submission(frame, expected, sample.columns.tolist(), labels, answer_kind=kind)
    if regression and not problems:
        values = pd.to_numeric(frame.drop(columns=identifier).iloc[:, 0], errors="coerce")
        if not np.isfinite(values).all() or (_nonnegative(task_spec) and (values < 0).any()):
            problems.append("invalid regression predictions")
    if problems:
        raise RuntimeError("Invalid submission: " + "; ".join(problems))
    if frame.drop(columns=identifier).nunique().max() == 1:
        print("Warning: constant submission", file=sys.stderr, flush=True)


if __name__ == "__main__":
    from leak_audit import LeakAuditError

    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    args = json.load(sys.stdin.buffer)
    output_directory.set(args.pop("output_dir", None))
    receipt_directory.set(args.pop("receipt_dir", None))
    try:
        if args["recipe"] == "ensemble":
            report = run_ensemble(Path.cwd(), args["task_spec"], time_budget=args["time_budget"],
                                  refit=bool(args["config"].get("refit")))
        elif args["recipe"] == "tune":
            report = run_prefix_tuning(Path.cwd(), args["task_spec"], args["config"], args["time_budget"])
        else:
            report = run_experiment(Path.cwd(), **args)
    except LeakAuditError as exc:
        report = {"skipped": f"leak_audit_failed: {exc}"}
    print(json.dumps(report, ensure_ascii=False, default=str), flush=True)
