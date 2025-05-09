import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
import joblib
from data.data_loader import CropDataLoader
from models.lightgbm_model import LightGBMModel
from visualization.plot_utils import (
    plot_feature_importance,
    plot_prediction_vs_actual,
    plot_correlation_matrix,
    plot_residuals,
)
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # 데이터 파일 경로 가져오기
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    else:
        data_file = "Crop_recommendationV2.csv"

    # 데이터 로드 및 전처리
    data_loader = CropDataLoader(data_file)
    df = data_loader.load_data()
    df = data_loader.preprocess()
    df = data_loader.create_growth_potential_score()

    # 특성과 타겟 분리
    target_col = "growth_potential"
    feature_cols = [
        col for col in df.columns if col not in [target_col, "label", "growth_stage"]
    ]

    X = df[feature_cols]
    y = df[target_col]

    # 학습/테스트 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 모델 학습
    model = LightGBMModel()

    # 하이퍼파라미터 최적화
    logger.info("하이퍼파라미터 최적화 시작...")
    best_params = model.optimize_hyperparameters(X_train, y_train, n_trials=20)

    # 최적의 하이퍼파라미터로 모델 학습
    logger.info("최적의 하이퍼파라미터로 모델 학습 시작...")
    model.train(X_train, y_train)

    # 모델 평가
    train_metrics = model.evaluate(X_train, y_train)
    test_metrics = model.evaluate(X_test, y_test)

    logger.info(f"학습 데이터 평가 결과: {train_metrics}")
    logger.info(f"테스트 데이터 평가 결과: {test_metrics}")

    # 예측값 생성
    y_pred = model.predict(X_test)

    # 시각화
    plot_feature_importance(model.model, feature_cols)
    plot_prediction_vs_actual(y_test, y_pred)
    plot_correlation_matrix(df[feature_cols + [target_col]])
    plot_residuals(y_test, y_pred)

    # 모델 저장
    model_path = Path("models/crop_growth_model.pkl")
    joblib.dump(model.model, model_path)
    logger.info(f"모델이 {model_path}에 저장되었습니다.")

    # MLflow에 결과 기록
    model.log_to_mlflow(test_metrics, best_params)


if __name__ == "__main__":
    main()
