import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import logging
import json
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_data():
    """테스트 데이터 생성"""
    data = {
        "N": 70,
        "P": 45,
        "K": 35,
        "temperature": 22,
        "humidity": 70,
        "ph": 6.5,
        "rainfall": 180,
        "soil_moisture": 20,
        "soil_type": 2,
        "sunlight_exposure": 8,
        "wind_speed": 10,
        "co2_concentration": 400,
        "organic_matter": 5,
        "irrigation_frequency": 3,
        "crop_density": 12,
        "pest_pressure": 50,
        "fertilizer_usage": 125,
        "growth_stage": 2,
        "urban_area_proximity": 25,
        "water_source_type": 2,
        "frost_risk": 50,
        "water_usage_efficiency": 3,
    }
    return pd.DataFrame([data])


def preprocess_data(df):
    """데이터 전처리"""
    # 수치형 특성 정규화
    numeric_features = [
        "N",
        "P",
        "K",
        "temperature",
        "humidity",
        "ph",
        "rainfall",
        "soil_moisture",
        "sunlight_exposure",
        "wind_speed",
        "co2_concentration",
        "organic_matter",
        "irrigation_frequency",
        "crop_density",
        "pest_pressure",
        "fertilizer_usage",
        "growth_stage",
        "urban_area_proximity",
        "frost_risk",
        "water_usage_efficiency",
    ]

    # 각 특성별 최소값과 최대값
    mins = {
        "N": 0,
        "P": 0,
        "K": 0,
        "temperature": 8.83,
        "humidity": 14.26,
        "ph": 3.50,
        "rainfall": 20.21,
        "soil_moisture": 10,
        "sunlight_exposure": 4,
        "wind_speed": 0,
        "co2_concentration": 300,
        "organic_matter": 0.5,
        "irrigation_frequency": 0,
        "crop_density": 4,
        "pest_pressure": 0,
        "fertilizer_usage": 50,
        "growth_stage": 1,
        "urban_area_proximity": 0,
        "frost_risk": 0,
        "water_usage_efficiency": 0.5,
    }

    maxs = {
        "N": 140,
        "P": 140,
        "K": 140,
        "temperature": 43.68,
        "humidity": 99.98,
        "ph": 9.93,
        "rainfall": 298.56,
        "soil_moisture": 40,
        "sunlight_exposure": 14,
        "wind_speed": 30,
        "co2_concentration": 500,
        "organic_matter": 10,
        "irrigation_frequency": 7,
        "crop_density": 20,
        "pest_pressure": 100,
        "fertilizer_usage": 200,
        "growth_stage": 3,
        "urban_area_proximity": 50,
        "frost_risk": 100,
        "water_usage_efficiency": 5,
    }

    # 수치형 특성 Min-Max 정규화
    for feature in numeric_features:
        min_val = mins[feature]
        max_val = maxs[feature]
        df[feature] = (df[feature] - min_val) / (max_val - min_val)

    # 범주형 특성 원-핫 인코딩
    categorical_features = ["soil_type", "water_source_type"]
    for feature in categorical_features:
        df = pd.get_dummies(df, columns=[feature], prefix=feature)

    # 특성 순서 맞추기
    expected_features = [
        "N",
        "P",
        "K",
        "temperature",
        "humidity",
        "ph",
        "rainfall",
        "soil_moisture",
        "sunlight_exposure",
        "wind_speed",
        "co2_concentration",
        "organic_matter",
        "irrigation_frequency",
        "crop_density",
        "pest_pressure",
        "fertilizer_usage",
        "growth_stage",
        "urban_area_proximity",
        "frost_risk",
        "water_usage_efficiency",
        "soil_type_1",
        "soil_type_2",
        "water_source_type_1",
    ]

    # 누락된 특성 추가
    for feature in expected_features:
        if feature not in df.columns:
            df[feature] = 0

    return df[expected_features]


def get_user_input():
    """사용자로부터 작물 데이터를 입력받습니다."""
    print("\n작물의 환경 조건을 입력해주세요:")

    data = {
        "N": float(input("질소(N) 함량 (0-140): ")),
        "P": float(input("인(P) 함량 (0-140): ")),
        "K": float(input("칼륨(K) 함량 (0-140): ")),
        "temperature": float(input("온도 (°C): ")),
        "humidity": float(input("습도 (%): ")),
        "ph": float(input("pH (0-14): ")),
        "rainfall": float(input("강수량 (mm): ")),
        "soil_moisture": float(input("토양 수분 (%): ")),
        "soil_type": int(input("토양 유형 (1: 사질토, 2: 양토, 3: 점토): ")),
        "sunlight_exposure": float(input("일조량 (시간/일): ")),
        "wind_speed": float(input("풍속 (km/h): ")),
        "co2_concentration": float(input("CO2 농도 (ppm): ")),
        "organic_matter": float(input("유기물 함량 (%): ")),
        "irrigation_frequency": int(input("관개 빈도 (회/주): ")),
        "crop_density": float(input("작물 밀도 (식물/m²): ")),
        "pest_pressure": float(input("해충 압력 (0-100): ")),
        "fertilizer_usage": float(input("비료 사용량 (kg/ha): ")),
        "growth_stage": int(input("생장 단계 (1: 초기, 2: 중기, 3: 후기): ")),
        "urban_area_proximity": float(input("도시 지역 근접도 (km): ")),
        "water_source_type": int(input("수원 유형 (1: 지하수, 2: 강수, 3: 관개수): ")),
        "frost_risk": float(input("서리 위험도 (0-100): ")),
        "water_usage_efficiency": float(input("수분 이용 효율 (0-5): ")),
    }

    return pd.DataFrame([data])


def normalize_prediction(prediction):
    """예측값을 0-100 범위로 정규화"""
    # 예측값을 0-100 범위로 변환
    normalized = prediction * 100
    return np.clip(normalized, 0, 100)


def evaluate_growth_potential(potential_score):
    """생장 잠재력 평가"""
    # 정규화된 점수(0-100)에 대한 평가
    if potential_score >= 70:
        return "매우 높음", "작물 생장에 매우 유리한 조건입니다."
    elif potential_score >= 50:
        return "높음", "작물 생장에 유리한 조건입니다."
    elif potential_score >= 30:
        return "보통", "작물 생장에 적절한 조건입니다."
    elif potential_score >= 10:
        return "낮음", "작물 생장에 불리한 조건입니다."
    else:
        return "매우 낮음", "작물 생장에 매우 불리한 조건입니다."


def load_data_from_json(json_file):
    """JSON 파일에서 데이터 로드"""
    try:
        with open(json_file, "r") as f:
            data = json.load(f)
        # daily_conditions의 첫 번째 날짜의 조건을 사용
        conditions = data["daily_conditions"][0]["conditions"]
        return pd.DataFrame([conditions])
    except Exception as e:
        logger.error(f"JSON 파일 로드 중 오류 발생: {str(e)}")
        raise


def main():
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description="작물 생장 잠재력 예측")
    parser.add_argument("--json", type=str, help="입력 데이터 JSON 파일 경로")
    args = parser.parse_args()

    # 모델 로드
    model_path = Path("models/crop_growth_model.pkl")
    if not model_path.exists():
        logger.error("모델 파일을 찾을 수 없습니다.")
        return

    model = joblib.load(model_path)
    logger.info("모델 로드 완료")

    while True:
        try:
            # 데이터 로드
            if args.json:
                test_df = load_data_from_json(args.json)
                logger.info(f"JSON 파일에서 데이터 로드 완료: {args.json}")
            else:
                test_df = create_test_data()
                logger.info("테스트 데이터 생성 완료")

            # 데이터 전처리
            processed_df = preprocess_data(test_df)
            logger.info("데이터 전처리 완료")

            # 예측
            predictions = model.predict(processed_df)
            logger.info("예측 완료")

            # 예측값 정규화 (0-100 범위로 변환)
            normalized_score = normalize_prediction(predictions[0])

            # 생장 잠재력 평가
            rating, description = evaluate_growth_potential(normalized_score)

            # 결과 출력
            result = {
                "raw_score": float(predictions[0]),
                "normalized_score": float(normalized_score),
                "rating": rating,
                "description": description,
            }

            print("\n=== 작물 생장 잠재력 평가 결과 ===")
            print(f"원시 점수: {result['raw_score']:.2f}")
            print(f"정규화된 점수: {result['normalized_score']:.2f}%")
            print(f"등급: {result['rating']}")
            print(f"설명: {result['description']}")

            # 결과를 CSV 파일에 저장
            results_df = pd.DataFrame([result])
            results_df.to_csv("predictions.csv", index=False)
            logger.info("예측 결과가 predictions.csv에 저장되었습니다.")

            if not args.json:
                continue_input = input(
                    "\n다른 작물의 생장 잠재력을 평가하시겠습니까? (y/n): "
                )
                if continue_input.lower() != "y":
                    break
            else:
                break

        except KeyboardInterrupt:
            print("\n프로그램을 종료합니다.")
            break
        except Exception as e:
            logger.error(f"오류 발생: {str(e)}")
            break


if __name__ == "__main__":
    main()
