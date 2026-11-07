import csv
import json
from io import StringIO
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


APP_DIR = Path(__file__).parent


@st.cache_resource(show_spinner=False)
def load_model(path):
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def load_test(path):
    return pd.read_csv(path)


@st.cache_data(show_spinner=False, max_entries=16)
def predict_probabilities(_model, X):
    return _model.predict_proba(X)[:, 1]


@st.cache_data(show_spinner=False, max_entries=4)
def read_uploaded_csv(content, feature_names):
    text = content.decode("utf-8-sig")
    header = next(csv.reader(StringIO(text)), [])
    if len(header) != len(set(header)):
        raise ValueError("В CSV повторяются названия столбцов.")
    frame = pd.read_csv(StringIO(text))
    missing = sorted(set(feature_names) - set(frame.columns))
    extra = sorted(set(frame.columns) - set(feature_names))
    errors = []
    if missing:
        errors.append("Не хватает столбцов: " + ", ".join(missing) + ".")
    if extra:
        errors.append("Лишние столбцы: " + ", ".join(extra) + ".")
    if errors:
        raise ValueError(" ".join(errors))
    if frame.empty:
        raise ValueError("В CSV нет строк с признаками вин.")
    frame = frame.loc[:, list(feature_names)].apply(pd.to_numeric, errors="coerce")
    invalid = frame.columns[frame.isna().any()].tolist()
    if invalid:
        raise ValueError("Пропущенные или нечисловые значения в столбцах: " + ", ".join(invalid) + ".")
    nonfinite = frame.columns[~np.isfinite(frame.to_numpy()).all(axis=0)].tolist()
    if nonfinite:
        raise ValueError("Бесконечные значения в столбцах: " + ", ".join(nonfinite) + ".")
    if not frame["is_red"].isin([0, 1]).all():
        raise ValueError("Столбец is_red должен содержать только 0 (белое) или 1 (красное).")
    return frame


def show_single_wine(model, test, feature_names, threshold):
    st.write("Введите признаки вина. Начальные значения — медианы тестовой выборки.")
    values = {}
    columns = st.columns(3)
    continuous_features = [name for name in feature_names if name != "is_red"]
    for index, feature in enumerate(continuous_features):
        lower = float(test[feature].min())
        upper = float(test[feature].max())
        with columns[index % 3]:
            values[feature] = st.number_input(
                feature,
                min_value=lower,
                max_value=upper,
                value=float(test[feature].median()),
                step=max((upper - lower) / 1000, 0.0001),
                format="%.4f",
                key="wine_" + feature,
            )
    color = st.radio(
        "Цвет вина",
        ["Белое", "Красное"],
        index=int(test["is_red"].median() >= 0.5),
        horizontal=True,
    )
    values["is_red"] = int(color == "Красное")
    wine = pd.DataFrame([values], columns=feature_names)
    probability = float(predict_probabilities(model, wine)[0])
    st.metric("Вероятность хорошего вина", f"{probability:.1%}")
    if probability >= threshold:
        st.success("Хорошее вино: включить в подборку.")
    else:
        st.info("Не включать в подборку при текущем пороге.")


def show_metrics(target, probabilities, threshold):
    prediction = (probabilities >= threshold).astype(int)
    precision = precision_score(target, prediction, zero_division=0)
    recall = recall_score(target, prediction, zero_division=0)
    f1 = f1_score(target, prediction, zero_division=0)
    columns = st.columns(3)
    for column, label, value in zip(columns, ["Precision", "Recall", "F1"], [precision, recall, f1]):
        column.metric(label, f"{value:.3f}")
    st.caption(
        f"Тест: {len(target)} вин. ROC AUC = {roc_auc_score(target, probabilities):.3f}; "
        f"Average precision = {average_precision_score(target, probabilities):.3f}."
    )

    matrix = confusion_matrix(target, prediction, labels=[0, 1])
    figure, axis = plt.subplots(figsize=(5, 3.5))
    axis.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > matrix.max() / 2 else "black",
            )
    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Нехорошее", "Хорошее"],
        yticklabels=["Нехорошее", "Хорошее"],
        xlabel="Предсказанный класс",
        ylabel="Истинный класс",
        title="Матрица ошибок",
    )
    figure.tight_layout()
    st.pyplot(figure)
    plt.close(figure)

    fpr, tpr, _ = roc_curve(target, probabilities)
    curve_precision, curve_recall, _ = precision_recall_curve(target, probabilities)
    true_negative, false_positive, _, _ = matrix.ravel()
    selected_fpr = false_positive / (false_positive + true_negative)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(fpr, tpr)
    axes[0].plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    axes[0].scatter(selected_fpr, recall, color="darkred", zorder=3, label=f"Порог {threshold:.3f}")
    axes[0].set(xlabel="False positive rate", ylabel="Recall", title="ROC-кривая")
    axes[1].plot(curve_recall, curve_precision)
    axes[1].axhline(float(np.mean(target)), linestyle="--", color="grey", linewidth=1)
    axes[1].scatter(recall, precision, color="darkred", zorder=3, label=f"Порог {threshold:.3f}")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="PR-кривая")
    for axis in axes:
        axis.set_xlim(-0.02, 1.02)
        axis.set_ylim(-0.02, 1.02)
        axis.legend()
    figure.tight_layout()
    st.pyplot(figure)
    plt.close(figure)
    st.write(
        "Более высокий порог сужает подборку и уменьшает полноту: "
        "больше хороших вин останется за её пределами. Более низкий порог "
        "расширяет подборку, но может добавить больше вин невысокого качества."
    )
    if not prediction.any():
        st.caption("При этом пороге в подборке нет вин; precision принят равным 0.")


def show_batch(model, test, feature_names, threshold):
    st.write("Загрузите CSV в UTF-8 с разделителем-запятой и 12 признаками, без target.")
    st.caption("is_red: 0 — белое, 1 — красное. Названия и порядок признаков видны в примере.")
    st.download_button(
        "Скачать пример CSV",
        test.loc[:, feature_names].head(3).to_csv(index=False).encode("utf-8"),
        file_name="wine_example.csv",
        mime="text/csv",
    )
    uploaded = st.file_uploader("CSV с признаками вин", type=["csv"])
    if uploaded is None:
        return
    try:
        frame = read_uploaded_csv(uploaded.getvalue(), tuple(feature_names))
    except (ValueError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        st.error(f"Не удалось прочитать CSV: {error}")
        return
    probabilities = predict_probabilities(model, frame)
    result = frame.copy()
    result["probability"] = probabilities
    result["prediction"] = (probabilities >= threshold).astype(int)
    st.caption(f"Порог {threshold:.3f}. prediction = 1: включить в подборку; 0: не включать.")
    st.dataframe(result, hide_index=True)
    st.download_button(
        "Скачать предсказания",
        result.to_csv(index=False).encode("utf-8"),
        file_name="wine_predictions.csv",
        mime="text/csv",
    )


def show_coefficients(model, feature_names):
    estimator = model.steps[-1][1]
    coefficients = pd.Series(estimator.coef_[0], index=feature_names).sort_values()
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.barh(
        coefficients.index,
        coefficients.values,
        color=["seagreen" if value >= 0 else "indianred" for value in coefficients],
    )
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set(xlabel="Коэффициент после масштабирования", title="Коэффициенты логистической регрессии")
    figure.tight_layout()
    st.pyplot(figure)
    plt.close(figure)
    positive = coefficients[coefficients > 0].nlargest(3)
    negative = coefficients[coefficients < 0].nsmallest(3)
    st.write("Сильнее всего за хорошее вино: " + ", ".join(f"{name} ({value:+.3f})" for name, value in positive.items()) + ".")
    st.write("Сильнее всего против: " + ", ".join(f"{name} ({value:+.3f})" for name, value in negative.items()) + ".")
    st.caption(
        "Вес показывает изменение логарифма шансов хорошего вина при увеличении "
        "признака на одно стандартное отклонение и неизменных остальных признаках."
    )


def main():
    st.set_page_config(page_title="Качество вина", layout="wide")
    st.title("Качество вина")
    st.write("Хорошее вино — оценка качества не ниже 7.")
    try:
        model = load_model(APP_DIR / "model.joblib")
        test = load_test(APP_DIR / "test.csv")
        config = json.loads((APP_DIR / "config.json").read_text(encoding="utf-8"))
        feature_names = config.get("feature_names", test.columns.drop("target").tolist())
        default_threshold = float(config["threshold"])
        if not 0 <= default_threshold <= 1:
            raise ValueError("Порог в config.json должен лежать от 0 до 1.")
        features = test.loc[:, feature_names]
        target = test["target"].to_numpy(dtype=int)
        probabilities = predict_probabilities(model, features)
    except (OSError, ValueError, KeyError) as error:
        st.error(f"Не удалось загрузить модель или данные: {error}")
        st.stop()

    threshold = st.sidebar.slider(
        "Порог хорошего вина",
        min_value=0.0,
        max_value=1.0,
        value=default_threshold,
        step=0.001,
        format="%.3f",
        key="threshold",
    )
    st.sidebar.caption("Один порог для всех вкладок. Вино выбирается, если вероятность не меньше порога.")
    single_tab, metrics_tab, batch_tab, coefficients_tab = st.tabs(
        ["Одно вино", "Модельные метрики", "Пакетное предсказание", "Что выучила модель"]
    )
    with single_tab:
        show_single_wine(model, test, feature_names, threshold)
    with metrics_tab:
        show_metrics(target, probabilities, threshold)
    with batch_tab:
        show_batch(model, test, feature_names, threshold)
    with coefficients_tab:
        show_coefficients(model, feature_names)


if __name__ == "__main__":
    main()
