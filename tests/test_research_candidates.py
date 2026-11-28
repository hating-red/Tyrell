import pipeline
import solution
import training


SPEC = {"audit_required": True, "task_type": "binary", "metric": "pr_auc",
        "available_backends": ["lightgbm", "catboost", "hgb", "linear"]}


def test_temporal_plan_only_queues_applicable_features_and_backends():
    control = {"backend": "lightgbm", "split": {"type": "time"},
               "feature_candidates": {"event_activity": True, "event_calendar": True, "interval_features": False}}
    queue = solution.choose_variant(SPEC, control)
    assert {row["model"] for row in queue} == set(SPEC["available_backends"])
    assert any(row.get("event_activity") for row in queue)
    assert any(row.get("event_calendar") for row in queue)
    assert all(not row.get("interval_features") and not row.get("causal_history") for row in queue)
    assert {"model": "catboost", "catboost_ordered_ctr": True} in queue
    delayed = solution.choose_variant(SPEC | {"label_delay_unit": "minutes"}, control)
    assert all(not row.get("catboost_ordered_ctr") for row in delayed)


def test_no_variant_without_a_baseline_or_an_available_backend():
    assert solution.choose_variant(SPEC, None) == []
    assert solution.choose_variant(SPEC | {"available_backends": []}, {"backend": "lightgbm"}) == []


def test_regularization_must_beat_calendar_incumbent_on_its_outer_folds():
    calendar = {"greater_is_better": False, "fold_scores": [4.3] * 5, "fold_signature": "outer-fits",
                "fold_baselines": [{"score": 6.}] * 5}
    for score, split, expected in ((4.2, "outer-fits", True), (4.4, "outer-fits", False), (4., "other", False)):
        regularized = calendar | {"fold_scores": [score] * 5, "fold_signature": split}
        assert pipeline._passes_quality_gate(calendar, regularized, training.compare_results) == expected


def test_ap_monitor_ablation_preserves_the_regularized_feature_bundle():
    baseline = {"backend": "catboost", "feature_candidates": {"event_activity": True, "causal_history": True}}
    regularized = {"model": "catboost", "auto_features": True, "event_activity": True, "causal_history": True,
                   "regularized_ap": True, "event_calendar": True}
    trials = solution._loss_variants(SPEC, baseline, accepted_features={"event_calendar"})
    assert regularized in trials and regularized | {"es_metric": "logloss"} in trials
