"""Тесты кодирования по таргету и модели на всем train. Запуск: python -m pytest tests -q"""

import numpy as np
import pandas as pd
import pytest

from features import add_key_columns, detect_entity_key, target_encoding_keys
from training import train_and_predict

N_ENTITIES, N_PERIODS = 400, 6


def entity_data(noise_target=False, seed=0):
    """Сущности (здания) повторяются по периодам; test - новый период тех же сущностей.

    Таргет почти целиком определяется скрытым «характером» сущности, которого нет в
    признаках, поэтому без кодирования по таргету модель его не узнает.
    """
    rng = np.random.default_rng(seed)
    area = rng.uniform(1_000, 100_000, N_ENTITIES).round()
    built = rng.integers(1900, 2020, N_ENTITIES)
    kind = rng.choice(["office", "school", "shop"], N_ENTITIES)
    hidden = rng.normal(0, 30, N_ENTITIES)
    rows = []
    for period in range(1, N_PERIODS + 2):
        weather = rng.normal(10, 5)
        for e in range(N_ENTITIES):
            rows.append({"period": period, "area": area[e], "built": built[e], "kind": kind[e],
                         "temp": weather + rng.normal(0, 0.5), "y": 100 + hidden[e] + 2 * weather})
    frame = pd.DataFrame(rows)
    if noise_target:
        frame["y"] = rng.normal(0, 10, len(frame))
    else:
        frame["y"] += rng.normal(0, 3, len(frame))
    train = frame[frame.period <= N_PERIODS].reset_index(drop=True)
    test = frame[frame.period == N_PERIODS + 1].reset_index(drop=True)
    return train.drop(columns="y"), train["y"], test.drop(columns="y")


def fit(X, y, X_test, target_encoding=False, **config):
    split = {"type": "group", "column": "period"}
    names = None
    if target_encoding:
        specs = target_encoding_keys(X, X_test, exclude=["period"], periods=X["period"])
        X, X_test, names = add_key_columns(X, X_test, specs)
    return train_and_predict(X, y, X_test, "regression", "rmse",
                             config={"split": split, "target_encoding": names, "num_boost_round": 300,
                                     **config})


def test_entity_key_uses_static_columns_only():
    X, _, X_test = entity_data()
    entity = detect_entity_key(X, X_test, exclude=["period"], periods=X["period"])
    assert entity is not None
    assert "temp" not in entity["columns"]          # меняется по периодам - не часть сущности
    assert set(entity["columns"]) & {"area", "built"}
    assert entity["coverage"] > 0.9


def test_no_entity_when_rows_do_not_repeat():
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"a": rng.normal(size=500), "b": rng.normal(size=500)})
    X_test = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200)})
    assert detect_entity_key(X, X_test) is None
    assert target_encoding_keys(X, X_test) == {}


def test_target_encoding_finds_hidden_entity_effect():
    X, y, X_test = entity_data()
    base = fit(X, y, X_test)
    with_te = fit(X, y, X_test, target_encoding=True)
    # Базовая модель частично узнает сущность и сама (area уникальна у каждой), поэтому
    # выигрыш не кратный, но заметный
    assert with_te["score"] < 0.8 * base["score"]
    assert with_te["config"]["target_encoding"] == ["__te_entity"]
    # Ключи в модель не идут, только их кодировки
    assert "__te_entity" not in with_te["config"]["features"]


def test_target_encoding_does_not_leak_on_noise():
    # Таргет - чистый шум: честное кодирование не может его предсказать. Утечка своего
    # таргета дала бы на CV заметно меньшую ошибку
    X, y, X_test = entity_data(noise_target=True)
    base = fit(X, y, X_test)
    with_te = fit(X, y, X_test, target_encoding=True)
    assert with_te["score"] > 0.97 * base["score"]
    assert with_te["score"] > 0.97 * float(np.std(y))


def test_refit_full_changes_test_but_not_cv():
    X, y, X_test = entity_data()
    folds_only = fit(X, y, X_test, target_encoding=True)
    refit = fit(X, y, X_test, target_encoding=True, refit_full=True)
    assert refit["refit"] is not None and refit["refit"]["applied"]
    assert refit["score"] == pytest.approx(folds_only["score"])
    assert not np.allclose(refit["predictions"], folds_only["predictions"])
    assert np.isfinite(refit["predictions"]).all()


def test_target_encoding_multiclass():
    X, y, X_test = entity_data()
    labels = pd.cut(y, 3, labels=["low", "mid", "high"]).astype(str)
    specs = target_encoding_keys(X, X_test, exclude=["period"], periods=X["period"])
    Xk, Xk_test, names = add_key_columns(X, X_test, specs)
    res = train_and_predict(Xk, labels, Xk_test, "multiclass", "macro_f1",
                            config={"split": {"type": "group", "column": "period"},
                                    "target_encoding": names, "num_boost_round": 200,
                                    "refit_full": True})
    assert np.isfinite(res["score"])
    assert res["probabilities"].shape == (len(X_test), 3)
    assert set(res["predictions"]) <= {"low", "mid", "high"}
