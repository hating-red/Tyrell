import json
import math
from pathlib import Path

import pandas as pd

from data_tools import load_data
from submission import make_submission, validate_submission


def run_experiment(task_dir, task_spec, config=None, time_budget=300):
    from training import METRICS, train_and_predict

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

    history_path = task_dir / ".tyrell_experiments.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    if history and history[0]["task_spec"] != task_spec:
        raise ValueError("Не меняйте тип задачи, target, ID или метрику между экспериментами.")
    cfg = {"num_boost_round": 300, "early_stopping_rounds": 30, **(config or {})}
    cfg.update(n_folds=3, fold_seed=42, n_jobs=8)
    cfg["params"] = dict(cfg.get("params") or {})
    for key in ("num_threads", "nthread", "thread_count", "n_jobs"):
        if key in cfg["params"]:
            cfg["params"][key] = 8
    result = train_and_predict(
        data["X_train"], data["y_train"], data["X_test"], task_type, metric,
        config=cfg, time_budget=time_budget,
    )
    if not math.isfinite(result["score"]):
        raise ValueError("Валидация вернула нечисловой результат.")
    saved = [item for item in history if item["saved"]]
    previous = saved[-1] if saved else None
    complete = result["config"]["n_folds_done"] == result["config"]["n_folds"]
    better = previous is None
    if previous is not None:
        same_folds = result["config"]["n_folds_done"] == previous["config"]["n_folds_done"]
        improved = (result["score"] > previous["score"] if result["greater_is_better"]
                    else result["score"] < previous["score"])
        better = (complete and not previous["complete"]) or (same_folds and improved)
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
    report = {
        "task_spec": task_spec, "score": result["score"], "metric": metric,
        "greater_is_better": result["greater_is_better"], "complete": complete,
        "saved": better and not problems, "problems": problems,
        "config": result["config"], "fit_time": result["fit_time"],
    }
    history.append(report)
    history_path.write_text(json.dumps(history, ensure_ascii=False), encoding="utf-8")
    return report


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
