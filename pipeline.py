import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from data_tools import detect_split, load_data
from submission import make_submission, validate_submission

# Ключи конфига, которые имеют смысл для всего рецепта auto, а не для одной модели
AUTO_KEYS = ("features", "cat_encoding")


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

    # Фолды одинаковые во всех экспериментах, иначе скоры несравнимы: агент их не меняет.
    # Разбиение группами включается само, если test похож на новый период
    split = detect_split(X_train, X_test)
    cfg = dict(config or {})
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
    problems = []
    if better:
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
        problems = make_submission(
            predictions, data["test_ids"], data["sample_path"], task_dir / "submission.csv",
            allowed_labels=labels, probabilities=probabilities, class_order=result["class_order"],
        )
    best_score = result["score"] if better and not problems else (previous["score"] if previous else None)
    report = {
        "task_spec": task_spec, "recipe": recipe, "feature_code": feature_code,
        "score": result["score"], "score_std": result["score_std"], "metric": metric,
        "fold_scores": [round(s, 6) for s in result.get("fold_scores", [])],
        "greater_is_better": result["greater_is_better"], "complete": complete,
        "n_folds_done": n_folds_done, "split": split,
        "saved": better and not problems, "problems": problems, "best_saved_score": best_score,
        "vs_saved": comparison and {k: comparison[k] for k in ("accept", "diff_mean", "diff_std", "n_better")},
        "members": result.get("members"),
        "config": {k: v for k, v in cfg.items() if k != "features"}, "fit_time": result["fit_time"],
    }
    history.append(report)
    history_path.write_text(json.dumps(history, ensure_ascii=False, default=str), encoding="utf-8")
    return report


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
