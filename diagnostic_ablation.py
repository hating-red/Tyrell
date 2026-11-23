"""Measure the marginal validation contribution of each applied PostgreSQL lookup."""
import json
import os
import tempfile
import time
from pathlib import Path


def run_ablations(task_dir, spec, control, deadline, experiment):
    """Repeat the first single-model recipe, removing one lookup at a time."""
    runtime_path = os.environ.get("TYRELL_RUNTIME_STATS")
    if not runtime_path or not Path(runtime_path).is_file():
        return
    joins = json.loads(Path(runtime_path).read_text(encoding="utf-8")).get("join_audit", [])
    output = Path(os.environ.get("TYRELL_OUTPUT_DIR", task_dir)) / "join-ablations.json"
    results = []
    report = {"control_recipe": "single", "metric": spec["metric"],
              "control_score": (control or {}).get("score"), "groups": results}

    def save():
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output.parent,
                                         suffix=".tmp", delete=False) as handle:
            temp = Path(handle.name)
            json.dump(report, handle, ensure_ascii=False, allow_nan=False, indent=2)
        try:
            os.replace(temp, output)
        finally:
            temp.unlink(missing_ok=True)

    for join in joins:
        applied = join["status"] == "applied"
        # Chained static dimensions have at most one parent hop. Removing the source
        # must remove features reached exclusively through that source as well.
        children = [other for other in joins if other["status"] == "applied" and
                    join["table"] in other.get("dependencies", [])]
        own = join.get("output_columns", [])
        excluded = list(dict.fromkeys(own + [col for child in children for col in child["output_columns"]]))
        results.append({"table": join["table"], "own_columns": own, "columns": excluded,
                        "dependent_tables": [item["table"] for item in children],
                        "status": "pending" if applied else "not_used",
                        "reason": None if applied else join.get("reason")})
    save()
    for group in sorted(results, key=lambda item: bool(item["dependent_tables"])):
        if group["status"] != "pending":
            continue
        if not control or not control.get("complete") or (control.get("split") or {}).get("type") != "time":
            group["status"] = "no_complete_temporal_control"
        elif not group["columns"]:
            group["status"] = "no_columns"
        else:
            remaining = deadline - time.monotonic()
            expected = max(3.0, float(control.get("training_seconds") or control.get("fit_time") or 0.0))
            if remaining < expected * 1.2 + 2:
                group["status"] = "insufficient_budget"
            else:
                config = {"exclude_columns": group["columns"], "diagnostic_group": group["table"]}
                start = time.monotonic()
                group["status"] = "running"
                save()
                candidate, raw = experiment(task_dir, spec, budget=min(remaining, 2 * expected + 8),
                                            config=config)
                group["seconds"] = round(time.monotonic() - start, 3)
                if not candidate:
                    group.update(status="failed", exit_code=raw.get("exit_code"), timeout=bool(raw.get("error")))
                elif (not candidate["complete"] or not control.get("fold_signature") or
                      candidate.get("fold_signature") != control["fold_signature"] or
                      candidate.get("membership_signature") != control.get("membership_signature") or
                      len(candidate["fold_scores"]) != len(control["fold_scores"])):
                    group["status"] = "incomparable_folds"
                else:
                    sign = 1 if control["greater_is_better"] else -1
                    group.update(status="measured", score_without=candidate["score"],
                                 fold_signature=control["fold_signature"],
                                 membership_signature=control.get("membership_signature"),
                                 gain_with_group=sign * (control["score"] - candidate["score"]),
                                 fold_gains_with_group=[round(sign * (a - b), 8) for a, b in
                                                        zip(control["fold_scores"], candidate["fold_scores"])])
        save()
