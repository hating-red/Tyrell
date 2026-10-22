"""Прогон train_and_predict на папке задачи.

Запуск: python run_real.py <папка задачи> <метрика> [--config JSON] [--task TYPE]
        [--budget СЕКУНДЫ] [--out results.jsonl]
id и таргет берутся из заголовка sample_submission.csv: первая колонка id, вторая таргет.
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from training import train_and_predict

logger = logging.getLogger("run_real")


def detect_task_type(y: pd.Series) -> str:
    """Тип задачи по таргету: нечисловой или целый с малым числом значений - классификация."""
    y = y.dropna()
    n_unique = y.nunique()
    is_numeric = pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y)
    if is_numeric and not (np.allclose(y, np.round(y)) and n_unique <= 20):
        return "regression"
    return "binary" if n_unique == 2 else "multiclass"


def load_task(task_dir: Path) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, str, str]:
    sample = pd.read_csv(task_dir / "sample_submission.csv", nrows=5)
    id_col, target_col = sample.columns[0], sample.columns[1]
    train = pd.read_csv(task_dir / "train.csv")
    test = pd.read_csv(task_dir / "test.csv")
    if target_col not in train.columns:
        raise ValueError(f"Таргет {target_col!r} из sample_submission не найден в train.csv")
    features = [c for c in train.columns if c not in (id_col, target_col) and c in test.columns]
    return train[features], train[target_col], test[features], id_col, target_col


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task_dir", type=Path)
    parser.add_argument("metric")
    parser.add_argument("--config", default=None, help="JSON с конфигом эксперимента")
    parser.add_argument("--task", default=None, help="regression, binary или multiclass")
    parser.add_argument("--budget", type=float, default=None, help="бюджет времени в секундах")
    parser.add_argument("--out", type=Path, default=None, help="куда дописать итог в формате jsonl")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    t0 = time.monotonic()
    X, y, X_test, id_col, target_col = load_task(args.task_dir)
    task_type = args.task or detect_task_type(y)
    config = json.loads(args.config) if args.config else None
    logger.info("Задача %s: таргет %s, тип %s, train %s, test %s, чтение %.1f с",
                args.task_dir.name, target_col, task_type, X.shape, X_test.shape, time.monotonic() - t0)

    res = train_and_predict(X, y, X_test, task_type, args.metric, config, time_budget=args.budget)

    summary = {
        "task": args.task_dir.name, "task_type": res["config"]["task_type"],
        "metric": res["config"]["metric"], "config": config or {},
        "score": res["score"], "score_std": res["score_std"], "score_raw": res["score_raw"],
        "fit_time": res["fit_time"], "fold_scores": res["fold_scores"],
        "best_iterations": res["best_iterations"],
    }
    if res["class_order"] is not None:
        pred_dist = pd.Series(res["predictions"]).value_counts(normalize=True)
        train_dist = y.value_counts(normalize=True)
        summary["pred_distribution"] = pred_dist.round(4).to_dict()
        summary["train_distribution"] = train_dist.round(4).to_dict()
        summary["postprocess"] = res["postprocess"]
        logger.info("Распределение классов, train и предсказания на test:\n%s",
                    pd.DataFrame({"train": train_dist, "test_pred": pred_dist}).fillna(0).round(4))
    else:
        p = np.asarray(res["predictions"])
        summary["pred_stats"] = {"mean": float(p.mean()), "min": float(p.min()), "max": float(p.max())}
        logger.info("Предсказания: mean %.3f, min %.3f, max %.3f; таргет train: mean %.3f",
                    p.mean(), p.min(), p.max(), y.mean())
    logger.info("Итог: %s", json.dumps(summary, ensure_ascii=False, default=str))

    if args.out:
        with args.out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(summary, ensure_ascii=False, default=str) + "\n")


if __name__ == "__main__":
    main()
