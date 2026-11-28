import json

import numpy as np
import pandas as pd
import pytest

import pipeline
import training


def _candidate(oof, test, fold_ids, y, *, model="lightgbm", temporal=False, known=None, metric="mae",
               target_bounds=None, integer=False):
    oof, test = np.asarray(oof, dtype=float), np.asarray(test, dtype=float)
    classification = oof.ndim == 2
    task_type = "binary" if classification else "regression"
    objective = training.get_metric(metric, task_type)
    scores = [training._score_fold(np.asarray(y)[fold_ids == k], oof[fold_ids == k], objective)
              for k in np.unique(fold_ids)]
    return {
        "oof": oof, "predictions": test.argmax(1) if classification else test,
        "probabilities": test if classification else None, "class_order": [0, 1] if classification else None,
        "fold_ids": fold_ids, "meta_known_from": known, "score": float(np.mean(scores)), "score_std": 0.,
        "fit_time": 1., "fold_scores": scores, "greater_is_better": objective.greater_is_better,
        "config": {"model": model, "task_type": task_type, "seed": 42, "n_folds": len(scores),
                   "split": {"type": "time", "column": "time"} if temporal else None,
                   "target_bounds": target_bounds, "integer_target": integer,
                   "fold_integer_target": [integer] * len(scores)},
    }


@pytest.mark.parametrize("model", ["hgb", "linear"])
def test_level1_backends_use_inner_monitors_on_real_temporal_data(model):
    n = 320
    positions = np.arange(n + 16)
    frame = pd.DataFrame({"event_time": pd.date_range("2024", periods=len(positions), freq="h"),
                          "x": np.sin(positions / 3),
                          "cat": [f"c{i % 68}" for i in positions]})
    y = 4. + 2 * frame["x"].iloc[:n]
    result = training.train_and_predict(
        frame.iloc[:n], y, frame.iloc[n:], "regression", "mae",
        {"model": model, "num_boost_round": 32, "early_stopping_rounds": 16, "n_jobs": 1,
         "audit_required": True, "label_delay_unit": "minutes"}, 20.,
    )
    assert result["config"]["model"] == model and result["config"]["n_folds_done"] == 5
    assert all(fold["fit_max"] < fold["monitor_min"] for fold in result["config"]["early_stopping_windows"])
    assert "event_time" not in result["model_inputs"]
    assert np.isfinite(result["predictions"]).all()


def test_linear_transform_uses_fit_and_past_targets():
    rng = np.random.default_rng(42)
    fit = pd.DataFrame({"key": [f"k{i % 36}" for i in range(108)], "small": ["a", "b"] * 54,
                        "x": rng.normal(size=108)})
    query = fit.iloc[:5].copy()
    clock = pd.Series(pd.date_range("2024", periods=len(fit), freq="h"))
    ask = clock.iloc[:5] + pd.Timedelta(days=8)
    with_clock = (clock, clock, ask, ask)
    labels = rng.normal(size=108)
    source, _, _ = training._linear_matrices((fit, query, query), labels, "regression", 0, with_clock)
    modified = query.assign(key="new", small="z", x=1e15)
    replay, _, _ = training._linear_matrices((fit, query, modified), labels, "regression", 0, with_clock)
    np.testing.assert_allclose(source.toarray(), replay.toarray())


def test_hgb_caps_native_category_dictionary_on_fit_rows():
    size = 350
    fit = pd.DataFrame({"key": pd.Categorical([f"k{i}" for i in range(size)]), "x": np.arange(size)})
    monitor = fit.iloc[:20]
    y = fit["x"].to_numpy(dtype=float)
    cfg = training._resolve_config({"model": "hgb", "n_jobs": 1, "num_boost_round": 2})
    _, prediction, *_ = training._fit_hgb(fit, y, None, monitor, y[:20], monitor, "regression",
                                          training.get_metric("mae", "regression"), 0, cfg, float("inf"))
    assert np.isfinite(prediction).all()


def test_temporal_stacker_weights_ignore_future_rows_and_delayed_labels():
    rng = np.random.default_rng(9)
    fid = np.repeat(np.arange(5), 50)
    y = 10 + rng.normal(size=len(fid))
    predictions = [y + rng.normal(size=len(y)), y + rng.normal(size=len(y)) * 2]
    known = fid + 1
    known[3] = 4
    members = [_candidate(p, [8., 12.], fid, y, temporal=True, known=known,
                          model=f"model{i}", target_bounds=(0., 20.)) for i, p in enumerate(predictions)]
    first = training.fit_causal_stacker(members, pd.Series(y), "mae")
    changed = [m | {"oof": m["oof"].copy()} for m in members]
    changed[0]["oof"][(fid >= 1) | (np.arange(len(y)) == 3)] += 1000
    second = training.fit_causal_stacker(changed, pd.Series(y), "mae")
    np.testing.assert_allclose(first["oof"][fid == 0], np.mean(predictions, axis=0)[fid == 0])
    assert first["config"]["fold_meta"][1] == second["config"]["fold_meta"][1]
    assert first["config"]["fold_meta"][1]["fit_rows"] == 49
    assert all(value >= 0 for value in first["config"]["fold_meta"][1]["weights"])


@pytest.mark.parametrize("classification", [False, True])
def test_dedup_keeps_best_cv_under_pearson_or_spearman(classification):
    monotone = np.linspace(.01, .99, 60)
    rng = np.random.default_rng(42)
    raw = np.stack([monotone, monotone ** 3 if classification else monotone * 2 + 1,
                    rng.permutation(monotone)])
    rows = [{"greater_is_better": classification, "config": {"model": name}}
            for name in ("catboost", "xgboost", "hgb")]
    if classification:
        raw = np.stack([1 - raw, raw], axis=2)
    keep, correlations, discarded = training.select_ensemble_pool(
        rows, raw, [1., 2., 3.] if classification else [2., 1., 3.], classification,
    )
    assert 1 in keep and 2 in keep and 0 not in keep
    assert correlations[0, 1] > .995 and discarded[0][:2] == (0, 1)


def test_rounding_is_cross_fitted_and_only_selected_for_integer_target():
    fid = np.repeat(np.arange(5), 10)
    y = np.tile(np.arange(10), 5)
    for integer in (True, False):
        labels = y if integer else y + .15
        single = _candidate(labels + .26, [2.26, 4.26], fid, labels, integer=integer, target_bounds=(0., 10.))
        best, _, _ = training.select_ensemble([single], pd.Series(labels), "mae")
        if integer:
            assert best["postprocess"]["type"] == "round"
            np.testing.assert_array_equal(best["predictions"], [2., 4.])
        else:
            assert not best.get("postprocess")


def test_auc_stays_uncalibrated_and_balanced_members_use_logit_in_first_fold():
    fid = np.repeat(np.arange(5), 20)
    y = np.tile([0, 1], 50)
    probabilities = np.linspace(.08, .89, len(y))
    members = []
    for i, p in enumerate((probabilities, np.roll(probabilities, 7))):
        member = _candidate(np.column_stack([1-p, p]), [[.5, .5]], fid, y, temporal=True,
                            metric="pr_auc", model=f"model{i}")
        if i:
            member["config"]["class_weight"] = "balanced"
        members.append(member)
    stack = training.fit_causal_stacker(members, pd.Series(y), "pr_auc")
    assert stack["config"]["fold_meta"][0]["space"] == "logit"
    best, _, stats = training.select_ensemble(members, pd.Series(y), "pr_auc")
    json.dumps(stats)
    assert not best.get("postprocess")
    assert all("postprocess" not in row for row in stats["strategies"])


def test_postprocessing_fits_temperature_and_class_weights_on_available_oof():
    p = np.tile([[.98, .02], [.98, .02], [.02, .98], [.02, .98]], (15, 1))
    y = np.tile([0, 1, 1, 0], 15)
    metric = training.get_metric("logloss", "binary")
    scale, info = training._fit_ensemble_postprocess(p, y, metric, False)
    calibrated = scale(p)
    assert info["type"] == "temperature" and 1. < info["temperature"] <= 5.
    assert metric.func(y, calibrated) < metric.func(y, p)
    np.testing.assert_allclose(calibrated.sum(axis=1), 1.)

    p = np.tile([[.9, .1], [.55, .45]], (30, 1))
    y = np.tile([0, 1], 30)
    metric = training.get_metric("f1", "binary")
    reweight, info = training._fit_ensemble_postprocess(p, y, metric, False)
    assert info["type"] == "class_weights"
    np.testing.assert_array_equal(reweight(p).argmax(1), y)


@pytest.mark.parametrize("metric", ["mae", "logloss"])
def test_serialized_fitted_meta_reproduces_test_predictions(metric):
    rng = np.random.default_rng(8)
    fid = np.repeat(np.arange(5), 30)
    classification = metric == "logloss"
    latent = rng.normal(size=len(fid))
    y = (latent > 0).astype(int) if classification else latent + 5
    members = []
    for i in range(2):
        oof = latent + rng.normal(size=len(latent)) * (i + 1)
        if classification:
            probability = 1 / (1 + np.exp(-oof))
            oof = np.column_stack([1 - probability, probability])
        else:
            oof += 5
        members.append(_candidate(oof, oof[:4], fid, y, model=f"model{i}", metric=metric,
                                  target_bounds=None if classification else (0., 10.)))
    fit = training.fit_causal_stacker(members, pd.Series(y), metric)
    meta = fit["config"]["final_meta"]
    key = "probabilities" if classification else "predictions"
    p = np.stack([members[i][key] for i in meta["members"]])
    replay = training.apply_ensemble_meta(meta, p, "binary" if classification else "regression", metric,
                                          bounds=None if classification else (0., 10.))
    np.testing.assert_allclose(replay, fit[key])


def test_ensemble_ap_gate_preserves_baseline_count_without_requiring_sweep(tmp_path):
    sample = tmp_path / "sample_submission.csv"
    pd.DataFrame({"id": ["a", "b"], "y": 0}).to_csv(sample, index=False)
    spec = {"task_type": "binary", "metric": "pr_auc", "audit_required": True, "probabilities": True}
    data = {"sample_path": sample, "test_ids": pd.Series(["a", "b"], name="id")}
    history = []
    control = {"score": .5, "score_std": 0., "fold_scores": [.56, .56, .46, .46, .46],
               "config": {"n_folds": 5, "fold_baselines": [{"score": .5}] * 5},
               "greater_is_better": True, "predictions": [0, 1], "probabilities": np.array([[.7, .3], [.2, .8]]),
               "class_order": [0, 1], "fold_ids": list(range(5)), "oof": [[.5, .5]] * 5, "fit_time": 1.}
    pipeline._record(tmp_path, spec, data, history, control, {"recipe": "single"})
    result = control | {"score": .53, "fold_scores": [.61, .61, .49, .47, .47]}
    saved = pipeline._record(tmp_path, spec, data, history, result, {"recipe": "ensemble"})
    assert saved["saved"]


def test_final_refit_selects_three_seeds_and_publishes_valid_ids(tmp_path):
    n = 100
    train = pd.DataFrame({"id": np.arange(n), "x": np.arange(n) / 10., "y": np.arange(n) / 3.})
    test = pd.DataFrame({"id": [300, 200, 150], "x": [2., 1., 5.]})
    train.to_csv(tmp_path / "train.csv", index=False)
    test.to_csv(tmp_path / "test.csv", index=False)
    test[["id"]].iloc[::-1].assign(y=0).to_csv(tmp_path / "sample_submission.csv", index=False)
    spec = {"target_column": "y", "id_columns": ["id"], "task_type": "regression", "metric": "mae",
            "audit_required": True}
    pipeline.run_experiment(tmp_path, spec, {"model": "linear"}, 20.)
    result = pipeline.run_ensemble(tmp_path, spec, time_budget=40., refit=True)
    assert result["saved"] and result["config"]["selected"] == [0]
    assert [fit["seed"] for fit in result["refit"]["fits"]] == [42, 43, 44]
    submission = pd.read_csv(tmp_path / "submission.csv")
    assert submission["id"].tolist() == [150, 200, 300]
    assert np.isfinite(submission["y"]).all()
