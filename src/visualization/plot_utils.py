import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_feature_importance(model, feature_names: List[str], top_n: int = 10):
    """특성 중요도 시각화"""
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance")
    plt.bar(range(top_n), importance[indices[:top_n]])
    plt.xticks(range(top_n), [feature_names[i] for i in indices[:top_n]], rotation=45)
    plt.tight_layout()
    plt.show()


def plot_prediction_vs_actual(
    y_true: np.ndarray, y_pred: np.ndarray, title: str = "Prediction vs Actual"
):
    """예측값과 실제값 비교 시각화"""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", lw=2)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_time_series(
    df: pd.DataFrame, target_col: str, feature_cols: Optional[List[str]] = None
):
    """시계열 데이터 시각화"""
    plt.figure(figsize=(15, 8))

    if feature_cols is None:
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in feature_cols:
        if col != target_col:
            plt.plot(df.index, df[col], label=col, alpha=0.5)

    plt.plot(df.index, df[target_col], label=target_col, linewidth=2)
    plt.legend()
    plt.title("Time Series Plot")
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(df: pd.DataFrame, method: str = "pearson"):
    """상관관계 행렬 시각화"""
    plt.figure(figsize=(12, 8))
    corr = df.corr(method=method)
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
    plt.title(f"Correlation Matrix ({method})")
    plt.tight_layout()
    plt.show()


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray):
    """잔차 시각화"""
    residuals = y_true - y_pred

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # 잔차 산점도
    ax1.scatter(y_pred, residuals, alpha=0.5)
    ax1.axhline(y=0, color="r", linestyle="--")
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Residuals")
    ax1.set_title("Residuals vs Predicted")

    # 잔차 히스토그램
    ax2.hist(residuals, bins=30)
    ax2.set_xlabel("Residuals")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Residuals Distribution")

    plt.tight_layout()
    plt.show()
