import pandas as pd
import numpy as np
from ml_models import CropGrowthPredictor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    cohen_kappa_score,
    balanced_accuracy_score,
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from typing import Tuple
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import PolynomialFeatures
import scipy.stats as stats


def load_test_data():
    """테스트 데이터 로드"""
    # CSV 파일에서 데이터 로드
    data = pd.read_csv("crop_growth_conditions_full.csv")
    return data


def plot_feature_importance(predictor, feature_names):
    """특성 중요도 시각화"""
    plt.figure(figsize=(12, 6))
    importance = predictor.base_models["random_forest"].feature_importances_
    indices = np.argsort(importance)[::-1]

    plt.title("특성 중요도")
    plt.bar(range(len(indices)), importance[indices])
    plt.xticks(
        range(len(indices)),
        [feature_names[i] for i in indices],
        rotation=45,
        ha="right",
    )
    plt.tight_layout()


def augment_data(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """데이터 증강"""
    # SMOTE로 데이터 증강
    smote = SMOTE(random_state=42)
    X_augmented, y_augmented = smote.fit_resample(X, y)

    return X_augmented, y_augmented


def create_polynomial_features(X: np.ndarray) -> np.ndarray:
    """다항 특성 생성"""
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    return X_poly


def preprocess_data(data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, list]:
    """데이터 전처리 함수"""
    # 작물 종류를 인코딩
    label_encoder = LabelEncoder()
    crop_encoded = label_encoder.fit_transform(data["crop"])

    # 성장 단계를 BBCH 스케일로 변환
    stage_mapping = {
        "initial": 10,
        "mid": 50,
        "late": 80,
    }
    y = data["growth_stage"].map(stage_mapping).values

    # 센서 데이터 및 환경 데이터 선택
    base_features = [
        "temperature",
        "humidity",
        "soil_moisture",
        "ph",
        "rainfall",
        "sunlight_exposure",
        "N",
        "P",
        "K",
        "co2_concentration",
    ]

    # 피처 컬럼 생성
    feature_columns = []
    feature_data = []

    for feature in base_features:
        # 기본 통계량
        stats_data = []
        for stat in ["avg", "min", "max"]:
            col = f"{feature}_{stat}"
            if col in data.columns:
                stats_data.append(data[col].values)
                feature_columns.append(col)

        if len(stats_data) > 0:
            stats_array = np.array(stats_data).T
            feature_data.append(stats_array)

            # 추가 통계량 계산
            if stats_array.shape[1] >= 2:
                # 표준편차
                std = np.std(stats_array, axis=1)
                feature_columns.append(f"{feature}_std")
                feature_data.append(std.reshape(-1, 1))

                # 범위
                range_val = np.ptp(stats_array, axis=1)
                feature_columns.append(f"{feature}_range")
                feature_data.append(range_val.reshape(-1, 1))

                # 변동계수 (CV)
                cv = np.std(stats_array, axis=1) / (
                    np.mean(stats_array, axis=1) + 1e-10
                )
                feature_columns.append(f"{feature}_cv")
                feature_data.append(cv.reshape(-1, 1))

    # 데이터 프레임 생성
    X = np.hstack(
        [arr if len(arr.shape) > 1 else arr.reshape(-1, 1) for arr in feature_data]
    )
    X = pd.DataFrame(X, columns=feature_columns)

    # 작물 종류를 피처로 추가
    X["crop"] = crop_encoded
    feature_columns.append("crop")

    # 다항 특성 생성 (2차까지만)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X.values)

    # 다항 특성의 컬럼명 생성
    poly_feature_names = []
    for i in range(X_poly.shape[1]):
        if i == 0:  # 상수항
            poly_feature_names.append("bias")
        else:
            poly_feature_names.append(f"poly_{i}")

    return X_poly, y, poly_feature_names


def evaluate_model(
    predictor: CropGrowthPredictor, X: np.ndarray, y: np.ndarray
) -> None:
    """모델 평가"""
    # 클래스 레이블 변환
    y_transformed = np.array([predictor.class_mapping.get(int(val), 0) for val in y])

    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_transformed, test_size=0.2, random_state=42, stratify=y_transformed
    )

    # 모델 학습
    results = predictor.train_models(X_train, y_train)

    # 예측
    predictions = predictor.predict_growth(X_test)

    # 예측 결과를 원래 클래스 레이블로 변환
    y_pred = []
    for pred in predictions:
        model_preds = []
        for model_name, pred_proba in pred.items():
            max_class = max(pred_proba.items(), key=lambda x: x[1])[0]
            model_preds.append(max_class)
        # 앙상블: 다수결 투표
        y_pred.append(max(set(model_preds), key=model_preds.count))

    # 평가 지표 계산
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    kappa = cohen_kappa_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print("\n=== 모델 성능 평가 ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Cohen's Kappa: {kappa:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)

    # 클래스별 성능
    print("\n클래스별 성능:")
    for i in range(len(np.unique(y_test))):
        class_name = {0: "초기", 1: "중기", 2: "후기"}[i]
        precision = (
            conf_matrix[i, i] / conf_matrix[:, i].sum()
            if conf_matrix[:, i].sum() > 0
            else 0
        )
        recall = (
            conf_matrix[i, i] / conf_matrix[i, :].sum()
            if conf_matrix[i, :].sum() > 0
            else 0
        )
        print(f"\n{class_name} 단계:")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")

    # 혼동 행렬 시각화
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["초기", "중기", "후기"],
        yticklabels=["초기", "중기", "후기"],
    )
    plt.xlabel("예측")
    plt.ylabel("실제")
    plt.title("혼동 행렬")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()


def test_model_pipeline():
    """전체 모델 파이프라인 테스트"""
    print("\n=== 작물 성장 예측 모델 테스트 ===")

    # 데이터 로드
    print("\n1. 데이터 로드 중...")
    data = pd.read_csv("crop_growth_conditions_full.csv")
    print(f"- 총 {len(data)} 개의 데이터 샘플")
    print(f"- 작물 종류: {', '.join(data['crop'].unique())}")
    print(f"- 성장 단계: {', '.join(data['growth_stage'].unique())}")

    # 데이터 전처리
    print("\n2. 데이터 전처리 중...")
    X, y, feature_names = preprocess_data(data)
    print(f"- 입력 특성 수: {X.shape[1]}")
    print(f"- 타겟 클래스: {np.unique(y)}")

    # 모델 초기화 및 학습
    print("\n3. 모델 학습 중...")
    predictor = CropGrowthPredictor()
    predictor.feature_names = feature_names
    results = predictor.train_models(X, y)

    # 모델 성능 출력
    print("\n=== 모델 성능 평가 ===")
    for name, scores in results.items():
        print(f"\n{name}:")
        print(
            f"- 교차 검증 Accuracy: {scores['mean_cv_accuracy']:.4f} (±{scores['std_cv_accuracy']:.4f})"
        )
        print(
            f"- 교차 검증 F1 Score: {scores['mean_cv_f1']:.4f} (±{scores['std_cv_f1']:.4f})"
        )
        print(
            f"- 각 폴드 Accuracy: {' '.join([f'{s:.4f}' for s in scores['cv_scores']])}"
        )
        print(
            f"- 각 폴드 F1 Score: {' '.join([f'{s:.4f}' for s in scores['cv_f1_scores']])}"
        )

    # 이상치 탐지
    print("\n4. 이상치 탐지 중...")
    X_processed, _ = predictor.preprocess_data(X, y)
    outlier_scores = predictor.detect_anomalies(X_processed)
    outlier_ratio = (outlier_scores < 0).mean()
    print(f"- 이상치 비율: {outlier_ratio:.2%}")
    print(f"- 평균 이상치 점수: {outlier_scores.mean():.4f}")
    print(
        f"- 최소/최대 이상치 점수: {outlier_scores.min():.4f} / {outlier_scores.max():.4f}"
    )

    # 예측 테스트
    print("\n5. 예측 테스트...")
    test_samples = X[:5]
    predictions = predictor.predict_growth(test_samples)

    print("\n=== 샘플 예측 결과 ===")
    for i, (pred, true_val) in enumerate(zip(predictions, y[:5]), 1):
        print(f"\n샘플 {i}:")
        print(f"실제 성장 단계: {true_val}")
        print(f"작물: {data['crop'].iloc[i-1]}")
        for model_name, pred_proba in pred.items():
            max_class = max(pred_proba.items(), key=lambda x: x[1])
            print(f"{model_name} 예측: {max_class[0]} (확률: {max_class[1]:.2f})")

    # 모델 저장 및 로드 테스트
    print("\n6. 모델 저장 및 로드 테스트...")
    predictor.save_models("crop_models.joblib")
    predictor.load_models("crop_models.joblib")
    print("모델 저장 및 로드 완료")

    # 모델 평가
    evaluate_model(predictor, X, y)


if __name__ == "__main__":
    test_model_pipeline()
