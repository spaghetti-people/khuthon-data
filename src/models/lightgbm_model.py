from lightgbm import LGBMRegressor
import optuna
from typing import Dict, Any
from .base_model import BaseModel


class LightGBMModel(BaseModel):
    def __init__(self):
        """LightGBM 모델 초기화"""
        super().__init__("lightgbm")

    def train(self, X, y):
        """모델 학습"""
        if self.best_params is None:
            self.model = LGBMRegressor()
        else:
            self.model = LGBMRegressor(**self.best_params)
        self.model.fit(X, y)

    def _get_trial_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Optuna trial에서 하이퍼파라미터 생성"""
        return {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1500),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        }

    def _create_model(self, params: Dict[str, Any]) -> LGBMRegressor:
        """하이퍼파라미터로 모델 생성"""
        return LGBMRegressor(**params)
