from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import sys
from pathlib import Path
import logging

# src 디렉토리를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))
from src.predict import preprocess_data, normalize_prediction
import joblib

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title="작물 생장 잠재력 예측 API",
    description="센서 데이터를 기반으로 작물의 생장 잠재력을 예측하는 API",
    version="1.0.0",
)

# 모델 로드
model_path = Path(__file__).parent / "models" / "crop_growth_model.pkl"
if not model_path.exists():
    raise RuntimeError("모델 파일을 찾을 수 없습니다.")

model = joblib.load(model_path)


class SensorData(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float
    soil_moisture: float
    soil_type: int
    sunlight_exposure: float
    wind_speed: float
    co2_concentration: float
    organic_matter: float
    irrigation_frequency: int
    crop_density: float
    pest_pressure: float
    fertilizer_usage: float
    growth_stage: int
    urban_area_proximity: float
    water_source_type: int
    frost_risk: float
    water_usage_efficiency: float


class PredictionResponse(BaseModel):
    growth_potential: float
    evaluation: str


@app.post("/predict", response_model=PredictionResponse)
async def predict_growth(sensor_data: SensorData):
    try:
        # 입력 데이터를 DataFrame으로 변환
        input_data = pd.DataFrame([sensor_data.dict()])

        # 데이터 전처리
        processed_data = preprocess_data(input_data)

        # 예측 수행
        prediction = model.predict(processed_data)[0]

        # 예측값 정규화 (0-100)
        normalized_prediction = normalize_prediction(prediction)

        # 생장 잠재력 평가
        evaluation = ""
        if normalized_prediction >= 80:
            evaluation = "매우 좋음"
        elif normalized_prediction >= 60:
            evaluation = "좋음"
        elif normalized_prediction >= 40:
            evaluation = "보통"
        elif normalized_prediction >= 20:
            evaluation = "나쁨"
        else:
            evaluation = "매우 나쁨"

        return PredictionResponse(
            growth_potential=float(normalized_prediction), evaluation=evaluation
        )

    except Exception as e:
        logger.error(f"예측 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail="예측 처리 중 오류가 발생했습니다.")


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
