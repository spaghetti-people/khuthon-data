import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import optuna
import mlflow
from typing import Dict, Any, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseModel:
    def __init__(self, model_name: str):
        """
        기본 모델 클래스 초기화

        Args:
            model_name (str): 모델 이름
        """
        self.model_name = model_name
        self.model = None
        self.best_params = None

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """모델 학습"""
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """예측 수행"""
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다.")
        return self.model.predict(X)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """모델 평가"""
        predictions = self.predict(X)
        metrics = {
            "mae": mean_absolute_error(y, predictions),
            "rmse": np.sqrt(mean_squared_error(y, predictions)),
            "r2": r2_score(y, predictions),
        }
        return metrics

    def optimize_hyperparameters(
        self, X: pd.DataFrame, y: pd.Series, n_trials: int = 20
    ) -> Dict[str, Any]:
        """하이퍼파라미터 최적화"""

        def objective(trial):
            params = self._get_trial_params(trial)
            model = self._create_model(params)

            # 시계열 교차 검증 - 데이터 크기에 맞게 조정
            n_splits = min(3, len(X) // 2)
            test_size = max(1, len(X) // 5)

            tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
            scores = []

            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                model.fit(X_train, y_train)
                pred = model.predict(X_val)
                score = mean_absolute_error(y_val, pred)
                scores.append(score)

            return np.mean(scores)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        self.best_params = study.best_params
        logger.info(f"최적 하이퍼파라미터: {self.best_params}")
        return self.best_params

    def _get_trial_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Optuna trial에서 하이퍼파라미터 생성"""
        raise NotImplementedError

    def _create_model(self, params: Dict[str, Any]) -> Any:
        """하이퍼파라미터로 모델 생성"""
        raise NotImplementedError

    def log_to_mlflow(
        self, metrics: Dict[str, float], params: Optional[Dict[str, Any]] = None
    ) -> None:
        """MLflow에 실험 결과 기록"""
        with mlflow.start_run(run_name=self.model_name):
            mlflow.log_metrics(metrics)
            if params:
                mlflow.log_params(params)
            if self.model:
                mlflow.sklearn.log_model(self.model, "model")
