import time
from unittest.mock import patch

import solution


def control(**fields):
    return dict(complete=True, fit_time=2., score=.1, greater_is_better=True, fold_scores=[.1] * 5,
                fold_signature="indices", membership_signature="train/validation", split={"type": "time"},
                feature_candidates={"entity_history": ["actor_id"], "categorical_ids": []}) | fields


def test_temporal_plan_checks_only_applicable_candidates_on_the_same_control(tmp_path):
    base = control()
    tested = []

    def run(task_dir, spec, budget, config, deadline):
        tested.append(config)
        assert budget <= deadline - time.monotonic() + .1
        if config.get("auto_features"):
            return control(score=.5, fold_scores=[.5] * 5, membership_signature="different"), {}
        if config.get("entity_history"):
            return control(score=.12, fold_scores=[.12] * 5), {}
        return base, {}

    with patch.object(solution, "experiment", side_effect=run):
        cfg = solution.choose_variant(tmp_path, {"task_type": "binary"}, base, time.monotonic()+50)
    assert cfg == {"entity_history": True}
    assert tested[0] == cfg and all("categorical_ids" not in x for x in tested)


def test_no_variant_after_research_slot_expires(tmp_path):
    with patch.object(solution, "experiment") as fit:
        assert solution.choose_variant(tmp_path, {"task_type": "binary"}, control(), time.monotonic()) is None
        fit.assert_not_called()


def test_regularization_must_beat_calendar_incumbent():
    base = control(score=6., fold_scores=[6.] * 5, greater_is_better=False, feature_candidates={})
    calendar = {"auto_features": True}
    regularized = calendar | {"params": {"min_child_samples": 100, "lambda_l2": 3.0}}
    for candidate_mae, signature, expected in ((4.2, "train/validation", regularized),
                                                  (4.4, "train/validation", calendar),
                                                  (4.0, "other", calendar)):
        reports = [base | {"score": 4.3, "fold_scores": [4.3] * 5},
                   base | {"score": candidate_mae, "fold_scores": [candidate_mae] * 5,
                           "membership_signature": signature}]
        with patch.object(solution, "experiment", side_effect=[(r, {}) for r in reports]) as fit:
            config = solution.choose_variant(".", {"task_type": "regression", "metric": "mae"}, base,
                                             time.monotonic() + 50)
        assert [call.kwargs["config"] for call in fit.call_args_list] == [calendar, regularized]
        assert config == expected


def test_ap_monitor_preserves_winning_tree_parameters():
    base = control(feature_candidates={})
    leaf = {"num_leaves": 31, "min_child_samples": 10}
    tuned = base | {"score": .14, "fold_scores": [.14] * 5}
    improved = base | {"score": .15, "fold_scores": [.15] * 5}
    with patch.object(solution, "experiment", side_effect=[(base, {}), (tuned, {}), (improved, {})]) as fit:
        config = solution.choose_variant(".", {"task_type": "binary", "metric": "pr_auc"}, base,
                                         time.monotonic() + 50)
    assert config == {"params": leaf | {"metric": "average_precision"}}
    assert fit.call_args_list[-1].kwargs["config"] == config
