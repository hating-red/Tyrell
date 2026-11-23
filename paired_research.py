"""Budgeted, declarative hypothesis selection for paired temporal experiments."""
import csv
import hashlib
import json
import math
import os
from importlib.metadata import version
from pathlib import Path
import sys
import tempfile
import time


QWEN_ENV = ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL")
_CANDIDATES = {
    "shallow_xgb": {"model": "xgboost", "params": {"max_depth": 3, "min_child_weight": 12}},
    "regularized_lgb": {"model": "lightgbm", "params": {"num_leaves": 15, "min_child_samples": 80}},
    "frequency_lgb": {"model": "lightgbm", "cat_encoding": "frequency", "categorical_ids": False},
    "calendar_xgb": {"model": "xgboost", "auto_features": True, "params": {"max_depth": 4}},
    "calendar_lgb": {"model": "lightgbm", "auto_features": True, "params": {"num_leaves": 31, "lambda_l2": 5}},
    "regularized_xgb": {"model": "xgboost", "params": {"max_depth": 5, "reg_lambda": 10, "min_child_weight": 8}},
    "low_rate_lgb": {"model": "lightgbm", "params": {"learning_rate": 0.03, "num_leaves": 31}},
}
_PARAMETERS = {
    "lightgbm": {"num_leaves": ("int", 8, 127), "min_child_samples": ("int", 5, 256),
                 "lambda_l2": ("float", 0., 30.), "learning_rate": ("float", .015, .18),
                 "feature_fraction": ("float", .5, 1.)},
    "xgboost": {"max_depth": ("int", 2, 8), "min_child_weight": ("float", 1., 64.),
                "reg_lambda": ("float", 0., 30.), "eta": ("float", .015, .18),
                "colsample_bytree": ("float", .5, 1.)},
}
_MAX_TRIALS = 3


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _file_digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _protocol(task_dir, output, spec, history):
    # Check exact model inputs including physical row order and target contents locally.
    # Template order is diagnostic only; the ID check against test is performed in each arm.
    inputs = {name: _file_digest(task_dir / name) for name in ("train.csv", "test.csv", "data_dictionary.csv")
              if (task_dir / name).exists()}
    if not {"train.csv", "test.csv"} <= inputs.keys():
        raise ValueError("Missing paired train or test snapshot")
    with (task_dir / "sample_submission.csv").open(encoding="utf-8-sig", newline="") as f:
        inputs["template_columns"] = next(csv.reader(f))
    inputs["task"] = _file_digest(output / "task.md")
    root = Path(__file__).resolve().parent
    modules = ("solution.py", "data_tools.py", "diagnostic_ablation.py", "features.py", "paired_research.py",
               "pipeline.py", "postgres_loader.py", "submission.py", "time_budget.py", "training.py")
    runtime = {name: _file_digest(root / name) for name in modules}
    runtime["versions"] = {pkg: version(pkg) for pkg in
                           ("lightgbm", "xgboost", "optuna", "numpy", "pandas", "scikit-learn")}
    runtime["python"] = sys.version
    runtime["spec"] = spec
    runtime["threads"] = [os.environ.get(k) for k in
                          ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")]
    # Reject comparisons when budget-dependent common stages produced different starting models.
    shared = [{k: h.get(k) for k in ("recipe", "config", "auto_features", "exclude_columns", "fold_signature",
                                       "complete", "saved", "fold_scores", "diagnostic_group")}
              for h in history]
    return {"inputs": _digest(inputs), "implementation": _digest(runtime), "common_history": _digest(shared)}


def _pick(client, metric, task_type, options, previous, timeout):
    payload = {"metric": metric, "task_type": task_type, "candidates": options,
               "allowed_params": _PARAMETERS, "validation": previous}
    result = client.chat.completions.create(
        model=os.environ["OPENAI_MODEL"], timeout=timeout,
        messages=[{"role": "system", "content": "Choose the next candidate using the temporal CV results. "
                   "Return one JSON object with hypothesis (an exact candidate name) and optional params. "
                   "Propose numeric changes to test a hypothesis within allowed_params for the chosen backend. "
                   "Each parameter definition gives type, inclusive minimum and maximum."},
                  {"role": "user", "content": json.dumps(payload, allow_nan=False)}],
    )
    answer = json.loads(result.choices[0].message.content)
    if not isinstance(answer, dict) or set(answer) - {"hypothesis", "params"} or answer.get("hypothesis") not in options:
        raise ValueError("Unknown candidate or option")
    name = answer["hypothesis"]
    overrides = answer.get("params", {})
    rules = _PARAMETERS[options[name]["model"]]
    if not isinstance(overrides, dict) or set(overrides) - rules.keys():
        raise ValueError("Unknown model parameter")
    for key, value in overrides.items():
        kind, lower, upper = rules[key]
        if (type(value) not in (int, float) or not math.isfinite(value) or not lower <= value <= upper
                or (kind == "int" and type(value) is not int)):
            raise ValueError("Model parameter outside allowed range")
    return name, overrides


def run_paired_search(task_dir, spec, client, deadline, experiment):
    """Spend one reserved phase on selection from the same candidate set in each arm."""
    task_dir = Path(task_dir)
    arm = os.environ.get("TYRELL_PAIR_ARM")
    write_report = arm in ("qwen", "automl")
    output = Path(os.environ.get("TYRELL_OUTPUT_DIR", task_dir))
    start = time.monotonic()
    history = json.loads((task_dir / ".tyrell_experiments.json").read_text(encoding="utf-8"))
    previous = []
    numeric_params = {"learning_rate", "num_leaves", "min_child_samples", "lambda_l2", "max_depth",
                      "min_child_weight", "reg_lambda", "feature_fraction"}
    for h in history:
        if not h.get("complete"):
            continue
        cfg = h.get("config") or {}
        details = {"recipe": h["recipe"], "model": cfg.get("model", "lightgbm") if h["recipe"] == "single" else "selection",
                   "auto_features": bool(h.get("auto_features")),
                   "entity_history": bool(cfg.get("entity_history")), "categorical_ids": bool(cfg.get("categorical_ids")),
                   "lookup_removed": bool(h.get("diagnostic_group")), "score": h["score"], "fold_scores": h["fold_scores"]}
        details["params"] = {k: v for k, v in (cfg.get("params") or {}).items()
                             if k in numeric_params and isinstance(v, (int, float)) and math.isfinite(v)}
        previous.append(details)
    winner = next((h for h in reversed(history) if h.get("saved") and h["recipe"] != "ensemble"), {})
    base = {"auto_features": bool(winner.get("auto_features"))}
    for flag in ("entity_history", "categorical_ids"):
        if (winner.get("config") or {}).get(flag):
            base[flag] = True
    choices = {name: base | cfg for name, cfg in _CANDIDATES.items()}
    report = {"arm": arm, "metric": spec["metric"], "qwen_calls": 0, "attempted_calls": 0, "trials": [],
              "phase_budget_seconds": round(max(0, deadline - start), 2), "status": "running"}
    if write_report:
        report["protocol"] = _protocol(task_dir, output, spec, history)
        report["protocol"]["candidates"] = _digest({"candidates": choices, "allowed_params": _PARAMETERS})

    def save(status):
        report.update(status=status, elapsed_seconds=round(time.monotonic() - start, 2))
        if write_report:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output, delete=False) as f:
                path = Path(f.name)
                json.dump(report, f, ensure_ascii=False, allow_nan=False, indent=2)
            try:
                os.replace(path, output / "pair-search.json")
            finally:
                path.unlink(missing_ok=True)

    save("running")
    if arm == "qwen" and client is None:
        save("qwen_unavailable")
        return report
    average = next((h.get("training_seconds") for h in history if h.get("recipe") == "single" and h.get("complete")), None)
    estimated = max(4., float(average or 4.))
    for attempt in range(_MAX_TRIALS):
        left = deadline - time.monotonic()
        if left < 1.2 * estimated + 3:
            break
        if arm == "qwen":
            report["attempted_calls"] += 1
            try:
                candidate, overrides = _pick(client, spec["metric"], spec["task_type"], choices, previous,
                                  min(45., max(1., left - 1.2 * estimated - 2)))
            except Exception as error:
                report["error_type"] = type(error).__name__
                save("qwen_request_failed")
                return report
            report["qwen_calls"] += 1
        else:
            candidate, overrides = next(iter(choices)), {}
        cfg = choices.pop(candidate)
        if overrides:
            cfg = cfg | {"params": (cfg.get("params") or {}) | overrides}
        remaining = deadline - time.monotonic()
        if remaining < 1.2 * estimated + 1:
            break
        fit_budget = min(remaining, max(2.25 * estimated + 4, remaining / (_MAX_TRIALS - attempt)))
        result, _ = experiment(task_dir, spec, budget=fit_budget, config=cfg, deadline=deadline)
        complete = bool(result and result.get("complete"))
        trial = {"hypothesis": candidate, "config": cfg, "complete": complete, "score": result.get("score") if complete else None,
                 "fold_scores": result.get("fold_scores") if complete else None}
        report["trials"].append(trial)
        if complete:
            previous.append(trial)
        save("running")
    save("ok" if any(t["complete"] for t in report["trials"]) else "no_complete_trial")
    return report


def compare_arms(control, qwen):
    search = [r.get("paired_search") or {} for r in (control, qwen)]
    cv = [r.get("cv_audit") or {} for r in (control, qwen)]
    reasons = []
    for arm, r, s in zip(("automl", "qwen"), (control, qwen), search):
        if r.get("status") != "ok" or not r.get("csv_valid") or r.get("pair_arm") != arm:
            reasons.append(arm + "_run_failed")
        if s.get("status") != "ok" or s.get("arm") != arm or not any(t["complete"] for t in s.get("trials", [])):
            reasons.append(arm + "_search_incomplete")
    if not search[1].get("qwen_calls") or search[0].get("qwen_calls") != 0:
        reasons.append("qwen_activity_unverified")
    for key in ("metric", "task", "target", "time_limit_seconds", "budget_scope"):
        if not control.get(key) or control.get(key) != qwen.get(key):
            reasons.append(key + "_mismatch")
    # A large import, tunnel or snapshot delay in one arm can consume its common-stage
    # budget even when both were invoked with the very same nominal limit.
    for key in ("solution_budget_seconds", "remaining_for_research_seconds"):
        a, b = [r.get(key) or 0 for r in (control, qwen)]
        if min(a, b) <= 0 or abs(a - b) > max(2., .03 * min(a, b)):
            reasons.append(key + "_mismatch")
    for key in ("inputs", "implementation", "common_history", "candidates"):
        a, b = [(s.get("protocol") or {}).get(key) for s in search]
        if not a or a != b:
            reasons.append(key + "_mismatch")
    if not cv[0].get("membership_signature") or cv[0].get("membership_signature") != cv[1].get("membership_signature"):
        reasons.append("different_or_unverified_folds")
    budgets = [s.get("phase_budget_seconds") or 0 for s in search]
    if min(budgets) <= 0 or max(budgets) - min(budgets) > max(1., .01 * min(budgets)):
        reasons.append("different_search_budgets")
    scores = [r.get("temporal_cv_score") for r in (control, qwen)]
    if not all(isinstance(s, (int, float)) and math.isfinite(s) for s in scores):
        reasons.append("missing_score")
    paired = {"comparable": not reasons, "reasons": reasons, "metric": control.get("metric"),
              "automl_cv": scores[0], "qwen_arm_cv": scores[1], "qwen_calls": search[1].get("qwen_calls"),
              "search_seconds": [s.get("elapsed_seconds") for s in search], "gain": None, "fold_gains": None}
    if not reasons:
        greater = -1 if control["metric"] == "mae" else 1
        paired["gain"] = greater * (scores[1] - scores[0])
        paired["fold_gains"] = [greater * (b["model_score"] - a["model_score"]) for a, b in
                                zip(cv[0]["folds"], cv[1]["folds"], strict=True)]
    return paired


if __name__ == "__main__":
    paths = [Path(arg) for arg in sys.argv[1:]]
    print(json.dumps(compare_arms(*(json.loads(p.read_text(encoding="utf-8-sig")) for p in paths)), allow_nan=False))
