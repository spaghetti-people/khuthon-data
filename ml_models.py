import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest, VotingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.feature_selection import (
    SelectFromModel,
    mutual_info_classif,
    SelectKBest,
    VarianceThreshold,
)
from sklearn.decomposition import PCA
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from typing import Dict, List, Tuple, Union
import joblib
from scipy import stats
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    cohen_kappa_score,
    balanced_accuracy_score,
)
from imblearn.over_sampling import SMOTE


class CropGrowthPredictor:
    def __init__(self):
        # 기본 모델 설정
        self.base_models = {
            "random_forest": RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced",
                bootstrap=True,
                max_features="sqrt",
                criterion="entropy",
            ),
            "lightgbm": LGBMClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.01,
                num_leaves=31,
                random_state=42,
                verbose=-1,
                n_jobs=-1,
                lambda_l1=0.1,
                lambda_l2=0.1,
                objective="multiclass",
                metric="multi_logloss",
                num_class=3,
                class_weight="balanced",
                boosting_type="gbdt",
                feature_fraction=0.8,
                bagging_fraction=0.8,
                bagging_freq=5,
            ),
            "xgboost": XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.01,
                random_state=42,
                verbosity=0,
                n_jobs=-1,
                reg_alpha=0.1,
                reg_lambda=0.1,
                scale_pos_weight=1.0,
                num_class=3,
                objective="multi:softprob",
                tree_method="hist",
                grow_policy="lossguide",
                max_bin=256,
                subsample=0.8,
                colsample_bytree=0.8,
            ),
        }

        # 앙상블 모델 설정
        self.ensemble = VotingClassifier(
            estimators=[
                ("random_forest", self.base_models["random_forest"]),
                ("lightgbm", self.base_models["lightgbm"]),
                ("xgboost", self.base_models["xgboost"]),
            ],
            voting="soft",
            weights=[0.4, 0.3, 0.3],
            n_jobs=-1,
        )

        self.scaler = RobustScaler()
        self.label_encoder = LabelEncoder()
        self.feature_selector = None
        self.anomaly_detector = None
        self.feature_names = None
        self.selected_features = None
        self.pca = None
        self.variance_threshold = None
        self.correlation_threshold = 0.95  # 상관관계 임계값 증가
        self.n_components = 10  # PCA 컴포넌트 수 감소
        self.n_features_to_select = 20  # 선택할 특성 수 감소
        self.is_fitted = False
        self.to_drop = set()
        self.keep_indices = []

        # Ordinal Classification을 위한 클래스 매핑
        self.stage_mapping = {"initial": 10, "mid": 50, "late": 80}
        self.class_mapping = {
            10: 0,  # 초기 단계
            50: 1,  # 중기 단계
            80: 2,  # 후기 단계
        }
        self.reverse_class_mapping = {v: k for k, v in self.class_mapping.items()}

    def remove_correlated_features(
        self, X: np.ndarray, feature_names: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """상관관계가 높은 특성 제거"""
        if not self.is_fitted:
            # 학습 시에만 상관관계 계산
            corr_matrix = np.corrcoef(X.T)
            np.fill_diagonal(corr_matrix, 0)  # 대각선을 0으로 설정

            # 상관관계가 높은 특성 쌍 찾기
            high_corr_pairs = np.where(np.abs(corr_matrix) > self.correlation_threshold)
            high_corr_pairs = list(zip(high_corr_pairs[0], high_corr_pairs[1]))
            high_corr_pairs = [(i, j) for i, j in high_corr_pairs if i < j]  # 중복 제거

            # 제거할 특성 선택
            self.to_drop = set()
            for i, j in high_corr_pairs:
                if i not in self.to_drop and j not in self.to_drop:
                    # 분산이 더 작은 특성을 제거
                    var_i = np.var(X[:, i])
                    var_j = np.var(X[:, j])
                    self.to_drop.add(j if var_i > var_j else i)

            # 선택된 특성 인덱스 저장
            self.keep_indices = [i for i in range(X.shape[1]) if i not in self.to_drop]

        # 선택된 특성만 유지
        X_filtered = X[:, self.keep_indices]
        feature_names_filtered = [feature_names[i] for i in self.keep_indices]

        return X_filtered, feature_names_filtered

    def remove_low_variance_features(
        self, X: np.ndarray, feature_names: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """분산이 낮은 특성 제거"""
        self.variance_threshold = VarianceThreshold(threshold=0.01)
        X_var = self.variance_threshold.fit_transform(X)
        mask = self.variance_threshold.get_support()
        return X_var, [f for i, f in enumerate(feature_names) if mask[i]]

    def select_features_univariate(
        self, X: np.ndarray, y: np.ndarray, feature_names: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """단변량 특성 선택"""
        selector = SelectKBest(mutual_info_classif, k=self.n_features_to_select)
        X_selected = selector.fit_transform(X, y)
        mask = selector.get_support()
        return X_selected, [f for i, f in enumerate(feature_names) if mask[i]]

    def apply_pca(self, X: np.ndarray) -> np.ndarray:
        """PCA 적용"""
        if self.pca is None:
            self.pca = PCA(n_components=self.n_components)
            return self.pca.fit_transform(X)
        return self.pca.transform(X)

    def preprocess_data(
        self, X: np.ndarray, y: np.ndarray = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """데이터 전처리"""
        if y is not None:
            # 데이터 스케일링
            X_scaled = self.scaler.fit_transform(X)

            if not self.is_fitted:
                # 1. 분산이 낮은 특성 제거
                self.variance_threshold = VarianceThreshold(threshold=0.01)
                X_var = self.variance_threshold.fit_transform(X_scaled)
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]
                feature_names = [
                    feature_names[i]
                    for i in range(len(feature_names))
                    if self.variance_threshold.get_support()[i]
                ]
                print(f"분산 임계치 적용 후 특성 수: {X_var.shape[1]}")

                # 2. 상관관계가 높은 특성 제거
                X_corr, feature_names = self.remove_correlated_features(
                    X_var, feature_names
                )
                print(f"상관관계 제거 후 특성 수: {X_corr.shape[1]}")

                # 3. 단변량 특성 선택
                self.feature_selector = SelectKBest(
                    mutual_info_classif, k=min(20, X_corr.shape[1])
                )
                X_selected = self.feature_selector.fit_transform(X_corr, y)
                feature_names = [
                    feature_names[i]
                    for i in self.feature_selector.get_support(indices=True)
                ]
                print(f"단변량 선택 후 특성 수: {X_selected.shape[1]}")

                # 4. PCA 적용
                self.pca = PCA(n_components=min(10, X_selected.shape[1]), whiten=True)
                X_pca = self.pca.fit_transform(X_selected)
                print(f"PCA 적용 후 특성 수: {X_pca.shape[1]}")
                print(
                    f"PCA 설명된 분산 비율: {np.sum(self.pca.explained_variance_ratio_):.2f}"
                )

                # 5. 이상치 탐지 및 제거
                self.anomaly_detector = IsolationForest(
                    contamination=0.1,
                    random_state=42,
                    n_estimators=200,
                    max_samples="auto",
                    bootstrap=True,
                    n_jobs=-1,
                )
                outlier_scores = self.anomaly_detector.fit_predict(X_pca)
                is_inlier = outlier_scores == 1
                print(f"이상치 제거 후 샘플 수: {np.sum(is_inlier)}/{len(is_inlier)}")

                # 6. SMOTE로 데이터 불균형 해결
                smote = SMOTE(
                    random_state=42,
                    k_neighbors=min(5, np.bincount(y[is_inlier]).min() - 1),
                )
                X_balanced, y_balanced = smote.fit_resample(
                    X_pca[is_inlier], y[is_inlier]
                )
                print(f"SMOTE 적용 후 샘플 수: {len(y_balanced)}")
                print(f"클래스별 분포: {np.bincount(y_balanced)}")

                self.feature_names = feature_names
                self.is_fitted = True
                return X_balanced, y_balanced
            else:
                # 학습된 전처리 파이프라인 적용
                X_var = self.variance_threshold.transform(X_scaled)
                X_corr, _ = self.remove_correlated_features(
                    X_var, [f"feature_{i}" for i in range(X_var.shape[1])]
                )
                X_selected = self.feature_selector.transform(X_corr)
                X_pca = self.pca.transform(X_selected)
                return X_pca, y
        else:
            # 예측을 위한 전처리
            if not self.is_fitted:
                raise ValueError("모델이 학습되지 않았습니다.")
            X_scaled = self.scaler.transform(X)
            X_var = self.variance_threshold.transform(X_scaled)
            X_corr, _ = self.remove_correlated_features(
                X_var, [f"feature_{i}" for i in range(X_var.shape[1])]
            )
            X_selected = self.feature_selector.transform(X_corr)
            X_pca = self.pca.transform(X_selected)
            return X_pca

    def train_models(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """모델 학습"""
        # 클래스 레이블 변환
        y_transformed = np.array([self.class_mapping.get(int(val), 0) for val in y])

        # 데이터 전처리
        X_processed, y_processed = self.preprocess_data(X, y_transformed)

        # 학습/검증 데이터 분할
        X_train, X_val, y_train, y_val = train_test_split(
            X_processed,
            y_processed,
            test_size=0.2,
            random_state=42,
            stratify=y_processed,
        )

        # 교차 검증 설정
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        results = {}

        # 각 모델 학습 및 평가
        for name, model in self.base_models.items():
            try:
                # 교차 검증
                cv_scores = cross_val_score(
                    model, X_processed, y_processed, cv=cv, scoring="accuracy"
                )
                cv_f1_scores = cross_val_score(
                    model, X_processed, y_processed, cv=cv, scoring="f1_weighted"
                )
                cv_balanced_acc_scores = cross_val_score(
                    model, X_processed, y_processed, cv=cv, scoring="balanced_accuracy"
                )

                # 모델 학습
                if name == "lightgbm":
                    model.fit(
                        X_train,
                        y_train,
                        eval_set=[(X_val, y_val)],
                        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
                    )
                elif name == "xgboost":
                    model.fit(
                        X_train,
                        y_train,
                        eval_set=[(X_val, y_val)],
                        verbose=True,
                    )
                else:
                    model.fit(X_processed, y_processed)

                # 예측 및 추가 지표 계산
                y_pred = model.predict(X_processed)
                kappa = cohen_kappa_score(y_processed, y_pred)

                results[name] = {
                    "mean_cv_accuracy": cv_scores.mean(),
                    "std_cv_accuracy": cv_scores.std(),
                    "mean_cv_f1": cv_f1_scores.mean(),
                    "std_cv_f1": cv_f1_scores.std(),
                    "mean_cv_balanced_acc": cv_balanced_acc_scores.mean(),
                    "std_cv_balanced_acc": cv_balanced_acc_scores.std(),
                    "kappa": kappa,
                    "cv_scores": cv_scores,
                    "cv_f1_scores": cv_f1_scores,
                    "cv_balanced_acc_scores": cv_balanced_acc_scores,
                }
            except Exception as e:
                print(f"\n{name} 모델 학습 중 오류 발생: {str(e)}")
                continue

        # 앙상블 모델 학습
        try:
            self.ensemble.fit(X_processed, y_processed)
        except Exception as e:
            print(f"\n앙상블 모델 학습 중 오류 발생: {str(e)}")

        return results

    def detect_anomalies(self, X: np.ndarray) -> np.ndarray:
        """이상치 탐지"""
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다.")

        # 이미 전처리된 데이터를 받으므로 바로 이상치 탐지 수행
        return self.anomaly_detector.predict(X)

    def predict_growth(self, X: np.ndarray) -> List[Dict[str, np.ndarray]]:
        """성장 단계 예측"""
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다.")

        # 데이터 전처리
        X_processed = self.preprocess_data(X)

        # 각 모델의 예측 확률
        predictions = []
        for sample in X_processed:
            sample_pred = {}
            for name, model in self.base_models.items():
                try:
                    # 차원 확장
                    sample_reshaped = sample.reshape(1, -1)
                    pred_proba = model.predict_proba(sample_reshaped)[0]
                    # 예측 확률을 원래 클래스 레이블로 매핑
                    mapped_pred = {
                        self.reverse_class_mapping[i]: prob
                        for i, prob in enumerate(pred_proba)
                    }
                    sample_pred[name] = mapped_pred
                except Exception as e:
                    print(f"\n{name} 모델 예측 중 오류 발생: {str(e)}")
                    continue
            predictions.append(sample_pred)

        return predictions

    def save_models(self, path: str):
        """모델 저장"""
        model_data = {
            "base_models": self.base_models,
            "ensemble": self.ensemble,
            "scaler": self.scaler,
            "label_encoder": self.label_encoder,
            "feature_selector": self.feature_selector,
            "anomaly_detector": self.anomaly_detector,
            "feature_names": self.feature_names,
            "selected_features": self.selected_features,
            "class_mapping": self.class_mapping,
            "reverse_class_mapping": self.reverse_class_mapping,
        }
        joblib.dump(model_data, path)

    def load_models(self, path: str):
        """모델 로드"""
        model_data = joblib.load(path)
        self.base_models = model_data["base_models"]
        self.ensemble = model_data["ensemble"]
        self.scaler = model_data["scaler"]
        self.label_encoder = model_data["label_encoder"]
        self.feature_selector = model_data["feature_selector"]
        self.anomaly_detector = model_data["anomaly_detector"]
        self.feature_names = model_data["feature_names"]
        self.selected_features = model_data["selected_features"]
        self.class_mapping = model_data["class_mapping"]
        self.reverse_class_mapping = model_data["reverse_class_mapping"]

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """모델 학습"""
        # 데이터 전처리
        X_processed, y_processed = self.preprocess_data(X, y)

        # 모델 학습
        for name, model in self.base_models.items():
            model.fit(X_processed, y_processed)

        self.ensemble.fit(X_processed, y_processed)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """예측"""
        # 데이터 전처리
        X_processed, _ = self.preprocess_data(X)

        # 앙상블 모델로 예측
        return self.ensemble.predict(X_processed)
