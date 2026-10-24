import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from data_tools import detect_split, load_data
from submission import make_submission, validate_submission

PREDS_DIR = ".tyrell_preds"       # OOF и предсказания test завершенных экспериментов для ансамбля

# Ключи конфига, которые имеют смысл для всего рецепта auto, а не для одной модели
AUTO_KEYS = ("features", "cat_encoding", "target_encoding", "refit_full")


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
    result = make_features(full.copy())
    if not isinstance(result, pd.DataFrame) or len(result) != len(full):
        raise ValueError("make_features должна вернуть DataFrame с тем же числом строк.")
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

    history_path = task_dir / ".tyrell_experiments.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    if history and history[0]["task_spec"] != task_spec:
        raise ValueError("Не меняйте тип задачи, target, ID или метрику между экспериментами.")

    X_train, X_test = data["X_train"].reset_index(drop=True), data["X_test"].reset_index(drop=True)
    if feature_code:
        X_train, X_test = apply_feature_code(feature_code, X_train, X_test)
    y_train = data["y_train"].reset_index(drop=True)
    cfg = dict(config or {})
    auto_features = bool(cfg.pop("auto_features", False))
    if auto_features:
        from features import add_auto_features, important_numeric
        X_train, X_test = add_auto_features(X_train, X_test, important_numeric(X_train, y_train, task_type))

    # Фолды одинаковые во всех экспериментах, иначе скоры несравнимы: агент их не меняет.
    # Разбиение группами включается само, если test похож на новый период
    split = detect_split(data["X_train"], data["X_test"])

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

    if recipe == "auto":
        result = auto_train(
            X_train, y_train, X_test, task_type, metric, time_budget,
            config={k: cfg[k] for k in AUTO_KEYS if k in cfg} | fixed,
            max_seeds=max_seeds,
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
    return _record(task_dir, task_spec, data, history, result, report)


def _record(task_dir, task_spec, data, history, result, report):
    """Сравнение с сохраненным, запись сабмита при улучшении, предсказания для ансамбля, журнал."""
    from training import compare_results

    if not math.isfinite(result["score"]):
        raise ValueError("Валидация вернула нечисловой результат.")
    n_folds = result["config"]["n_folds"]
    n_folds_done = result["config"].get("n_folds_done", n_folds)
    saved = [item for item in history if item["saved"]]
    previous = saved[-1] if saved else None
    complete = n_folds_done == n_folds
    better = previous is None
    comparison = None
    if previous is not None:
        same_folds = n_folds_done == previous["n_folds_done"]
        if same_folds:
            # Не берем «везучую» модель: новый результат должен быть лучше на большинстве
            # фолдов и с приростом больше шума, а не просто чуть выше в среднем
            comparison = compare_results(
                {"fold_scores": previous["fold_scores"], "greater_is_better": previous["greater_is_better"]},
                {"fold_scores": result["fold_scores"], "greater_is_better": result["greater_is_better"]},
            )
        better = (complete and not previous["complete"]) or (same_folds and comparison["accept"])
    problems = _write_submission(task_dir, task_spec, data, result) if better else []

    # Предсказания завершенных экспериментов храним на диске: в конце их можно смешать
    preds_file = None
    if complete and "fold_ids" in result:
        preds_dir = task_dir / PREDS_DIR
        preds_dir.mkdir(exist_ok=True)
        preds_file = str(preds_dir / f"{len(history)}.npz")
        test = result["probabilities"] if result["probabilities"] is not None else result["predictions"]
        np.savez_compressed(preds_file, oof=np.asarray(result["oof"], dtype=np.float32),
                            test=np.asarray(test, dtype=np.float32), fold_ids=result["fold_ids"])

    report = report | {
        "score": result["score"], "score_std": result["score_std"], "metric": task_spec["metric"],
        "fold_scores": [round(s, 6) for s in result.get("fold_scores", [])],
        "greater_is_better": result["greater_is_better"], "complete": complete, "n_folds_done": n_folds_done,
        "saved": better and not problems, "problems": problems,
        "best_saved_score": result["score"] if better and not problems else (previous["score"] if previous else None),
        "vs_saved": comparison and {k: comparison[k] for k in ("accept", "diff_mean", "diff_std", "n_better")},
        "fit_time": result["fit_time"], "preds_file": preds_file, "class_order": result["class_order"],
    }
    history.append(report)
    (task_dir / ".tyrell_experiments.json").write_text(json.dumps(history, ensure_ascii=False, default=str),
                                                       encoding="utf-8")
    return report


def _write_submission(task_dir, task_spec, data, result):
    """Сабмит по формату sample_submission: метки, вероятность одного класса или всех классов."""
    task_type = task_spec["task_type"]
    probabilities = task_spec.get("probabilities", False)
    predictions = result["predictions"]
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
        predictions, data["test_ids"], data["sample_path"], Path(task_dir) / "submission.csv",
        allowed_labels=labels, probabilities=probabilities, class_order=result["class_order"],
    )


def run_ensemble(task_dir, task_spec, max_members=8):
    """Смесь лучших завершенных экспериментов: разные наборы признаков ошибаются по-разному.

    Фолды у всех экспериментов одинаковые, поэтому OOF смешиваются честно; веса подбираются
    с кросс-фиттингом, а сабмит заменяется по тому же правилу, что и после эксперимента.
    """
    from training import optimize_blend_weights

    task_dir = Path(task_dir)
    history_path = task_dir / ".tyrell_experiments.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
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
        stored = np.load(h["preds_file"])
        results.append({
            "oof": stored["oof"].astype(float), "fold_ids": stored["fold_ids"],
            "predictions": stored["test"].astype(float) if is_regression else None,
            "probabilities": None if is_regression else stored["test"].astype(float),
            "class_order": h["class_order"], "score": h["score"], "score_std": h["score_std"],
            "fold_scores": h["fold_scores"], "greater_is_better": greater, "fit_time": h["fit_time"],
            "config": {"task_type": task_spec["task_type"], "n_folds": h["n_folds_done"],
                       "n_folds_done": h["n_folds_done"], "seed": 42, "fold_seed": 42},
        })
    result = optimize_blend_weights(results, data["y_train"].reset_index(drop=True), task_spec["metric"])
    if result.get("predictions") is None and result.get("probabilities") is not None:
        # Лучшей оказалась одиночная модель из файла: метки восстанавливаем из вероятностей
        order = np.asarray(result["class_order"], dtype=object)
        result = result | {"predictions": order[np.asarray(result["probabilities"]).argmax(1)]}
    result.setdefault("config", {})
    result["config"] = result["config"] | {"n_folds": candidates[0]["n_folds_done"]}
    report = {"task_spec": task_spec, "recipe": "ensemble", "feature_code": None,
              "members": {f"exp{history.index(h)}": {"score": h["score"], "recipe": h["recipe"]} for h in candidates},
              "config": {"blend_weights": result["config"].get("blend_weights"),
                         "blend_type": result["config"].get("blend_type")}}
    return _record(task_dir, task_spec, data, history, result, report)


def write_fallback(task_dir, task_spec):
    """Простой ответ без модели: среднее таргета или самый частый класс. Только если файла еще нет."""
    task_dir = Path(task_dir)
    if (task_dir / "submission.csv").exists():
        return []
    data = load_data(task_dir, task_spec)
    y = data["y_train"].dropna()
    sample = pd.read_csv(data["sample_path"], nrows=0)
    n = len(data["test_ids"])
    if task_spec["task_type"] == "regression":
        predictions, probabilities, class_order, labels = np.full(n, float(y.mean())), False, None, None
    else:
        shares = y.value_counts(normalize=True).sort_index()
        class_order = shares.index.tolist()
        if task_spec.get("probabilities") or len(sample.columns) > 2:
            probabilities = True
            predictions = np.tile(shares.to_numpy(), (n, 1))
            if len(sample.columns) == 2:
                positive = task_spec.get("positive_class", class_order[-1])
                predictions = predictions[:, class_order.index(positive)]
            labels = None
        else:
            probabilities, labels = False, set(class_order)
            predictions = np.full(n, shares.idxmax(), dtype=object)
    return make_submission(
        predictions, data["test_ids"], data["sample_path"], task_dir / "submission.csv",
        allowed_labels=labels, probabilities=probabilities, class_order=class_order,
    )


def check_result(task_dir):
    task_dir = Path(task_dir)
    history = json.loads((task_dir / ".tyrell_experiments.json").read_text())
    saved = [item for item in history if item["saved"]]
    if not saved:
        raise RuntimeError("В этом запуске не был сохранён результат обучения.")
    spec = saved[-1]["task_spec"]
    data = load_data(task_dir, spec)
    frame = pd.read_csv(task_dir / "submission.csv", dtype={data["id_column"]: str})
    columns = pd.read_csv(data["sample_path"], nrows=0).columns.tolist()
    probabilities = spec.get("probabilities", False) or len(columns) > 2
    labels = set(data["y_train"].dropna()) if spec["task_type"] != "regression" and not probabilities else None
    problems = validate_submission(frame, data["test_ids"], columns, labels, probabilities)
    if problems:
        raise RuntimeError("Некорректный submission: " + "; ".join(problems))
