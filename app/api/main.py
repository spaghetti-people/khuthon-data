from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import sys
from pathlib import Path
import logging

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent))
from predict import preprocess_data, normalize_prediction
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

# ... 나머지 코드는 동일 ...
