"""Инструменты агента: загрузка, профиль данных, обучение с кросс-валидацией.

Каждая функция работает на любой таблице (не знает названий столбцов заранее)
и возвращает короткий словарь, который можно показать LLM как результат шага.
"""

from pathlib import Path

import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate

RANDOM_STATE = 42
N_FOLDS = 5

# Пороги эвристик. Вынесены наверх, чтобы их было легко найти и поменять.
MAX_CLASSES = 20  # целое число с ≤20 значениями считаем классом, иначе регрессия
TEXT_UNIQUE_RATIO = 0.5  # нечисловой столбец, где уникальных > 50% строк: текст или ID
LEAK_AUC = 0.9  # один признак сам по себе даёт ROC-AUC выше: подозрение на утечку
LEAK_R2 = 0.9  # то же для регрессии
LEAK_GROUP_SHARE = 0.05  # «чистая группа» должна занимать хотя бы 5% строк...
LEAK_GROUP_PURITY = 0.99  # ...и почти все строки в ней должны иметь один и тот же ответ


def load_data(path: str, na_values: list[str] | None = None) -> pd.DataFrame:
    """Читает csv или parquet. `na_values` — какие строки считать пропуском (например, "?")."""
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, na_values=na_values)


def detect_task(y: pd.Series) -> str:
    """Определяет тип задачи по целевой переменной: binary, multiclass или regression."""
    n_unique = y.nunique()
    if n_unique == 2:
        return "binary"
    is_integer_like = pd.api.types.is_integer_dtype(y) or not pd.api.types.is_numeric_dtype(y)
    if is_integer_like and n_unique <= MAX_CLASSES:
        return "multiclass"
    return "regression"


def prepare_features(df: pd.DataFrame, target: str, drop: list[str] | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Отделяет X и y, убирает столбцы из `drop`, переводит текстовые столбцы в category.

    Пропуски не заполняем: LightGBM обрабатывает их сам.
    """
    X = df.drop(columns=[target, *(drop or [])])
    y = df[target]
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].astype("category")
    return X, y


def _single_feature_score(x: pd.Series, y: pd.Series, task: str) -> float:
    """Насколько хорошо ОДИН признак сам по себе предсказывает ответ (быстрая модель, 3 фолда)."""
    X, _ = prepare_features(pd.DataFrame({"f": x, "y": y}), "y")
    if task == "regression":
        model, scoring, cv = LGBMRegressor(n_estimators=30, verbose=-1), "r2", KFold(3, shuffle=True, random_state=RANDOM_STATE)
    else:
        scoring = "roc_auc" if task == "binary" else "roc_auc_ovr"
        model, cv = LGBMClassifier(n_estimators=30, verbose=-1), StratifiedKFold(3, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring)
    return round(float(scores["test_score"].mean()), 3)


def _missing_pattern_leak(x: pd.Series, y: pd.Series) -> str | None:
    """Ищет «узкую» утечку: строки, где значение заполнено (или пустое), почти всегда имеют один ответ.

    Пример: номер тела есть только у погибших. Это 9% строк, поэтому общий ROC-AUC
    признака низкий, но про эти 9% он выдаёт ответ точно.
    """
    for label, mask in [("заполнено", x.notna()), ("пусто", x.isna())]:
        share = mask.mean()
        if share < LEAK_GROUP_SHARE:
            continue
        top_class_share = y[mask].value_counts(normalize=True).iloc[0]
        if top_class_share >= LEAK_GROUP_PURITY:
            top_class = y[mask].mode().iloc[0]
            return f"где значение {label} ({share:.0%} строк), ответ почти всегда {top_class}"
    return None


def profile_dataset(df: pd.DataFrame, target: str) -> dict:
    """Краткий профиль таблицы: размер, тип задачи, баланс классов, пропуски,
    столбцы-тексты/ID и признаки, подозрительные на утечку."""
    y = df[target]
    task = detect_task(y)
    features = [c for c in df.columns if c != target]

    missing = (df[features].isna().mean() * 100).round(1)
    text_or_id = [
        c for c in features
        if not pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > TEXT_UNIQUE_RATIO * len(df)
    ]

    leak_threshold = LEAK_R2 if task == "regression" else LEAK_AUC
    metric_name = "R2" if task == "regression" else "ROC-AUC"
    suspected_leaks = {}
    for col in features:
        if col in text_or_id:
            continue
        score = _single_feature_score(df[col], y, task)
        if score > leak_threshold:
            suspected_leaks[col] = f"один этот признак даёт {metric_name} {score}"
        elif task != "regression" and (reason := _missing_pattern_leak(df[col], y)):
            suspected_leaks[col] = reason

    profile = {
        "rows": len(df),
        "columns": len(features),
        "task": task,
        "missing_percent": {c: float(p) for c, p in missing[missing > 0].sort_values(ascending=False).items()},
        "text_or_id_columns": text_or_id,
        "suspected_leaks": suspected_leaks,
    }
    if task != "regression":
        profile["class_balance"] = {str(k): round(float(v), 3) for k, v in y.value_counts(normalize=True).items()}
    return profile


def _make_model(name: str, task: str):
    """Создаёт модель по имени. Список имён — это «меню», из которого выбирает агент."""
    is_clf = task != "regression"
    models = {
        "dummy": lambda: DummyClassifier(strategy="most_frequent") if is_clf else DummyRegressor(),
        "lgbm": lambda: (LGBMClassifier if is_clf else LGBMRegressor)(random_state=RANDOM_STATE, verbose=-1),
    }
    if name not in models:
        raise ValueError(f"Неизвестная модель '{name}'. Доступны: {list(models)}")
    return models[name]()


def train_and_evaluate(df: pd.DataFrame, target: str, model: str = "lgbm", drop: list[str] | None = None) -> dict:
    """Кросс-валидация выбранной модели. Возвращает средние метрики и их разброс между фолдами."""
    X, y = prepare_features(df, target, drop)
    task = detect_task(y)

    if task == "regression":
        cv = KFold(N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        scoring = {"rmse": "neg_root_mean_squared_error", "r2": "r2"}
        main_metric = "rmse"
    else:
        cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        if task == "binary":
            scoring = {"roc_auc": "roc_auc", "accuracy": "accuracy"}
            main_metric = "roc_auc"
        else:
            # Macro F1: среднее F1 по классам, редкие классы весят столько же, сколько частые
            scoring = {"f1_macro": "f1_macro", "accuracy": "accuracy"}
            main_metric = "f1_macro"

    scores = cross_validate(_make_model(model, task), X, y, cv=cv, scoring=scoring)

    metrics = {}
    for name in scoring:
        values = scores[f"test_{name}"]
        if name == "rmse":
            values = -values  # sklearn возвращает RMSE со знаком минус, чтобы «больше = лучше»
        metrics[name] = round(float(values.mean()), 4)
    metrics[f"{main_metric}_std"] = round(float(scores[f"test_{main_metric}"].std()), 4)  # разброс между фолдами

    return {
        "model": model,
        "task": task,
        "features_used": len(X.columns),
        "dropped": drop or [],
        "main_metric": main_metric,
        "metrics": metrics,
    }


def read_roles(dictionary_path: str) -> dict[str, list[str]]:
    """Читает data_dictionary.csv и группирует столбцы по роли: {"id": [...], "target": [...], "feature": [...]}."""
    dictionary = pd.read_csv(dictionary_path)
    return dictionary.groupby("role")["feature"].apply(list).to_dict()


if __name__ == "__main__":
    # Проверка на задаче кейса: python -m agent.tools data/tasks/regression
    import json
    import sys

    task_dir = Path(sys.argv[1])
    data_dir = next(task_dir.glob("**/participants"))
    roles = read_roles(data_dir / "data_dictionary.csv")
    target, ids = roles["target"][0], roles.get("id", [])

    df = load_data(data_dir / "train.csv")
    profile = profile_dataset(df.drop(columns=ids), target)
    print("ПРОФИЛЬ:", json.dumps(profile, ensure_ascii=False, indent=2))

    drop = ids + profile["text_or_id_columns"] + list(profile["suspected_leaks"])
    for model in ["dummy", "lgbm"]:
        print(json.dumps(train_and_evaluate(df, target, model, drop), ensure_ascii=False))
