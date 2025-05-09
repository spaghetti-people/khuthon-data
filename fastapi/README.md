# 작물 생장 잠재력 예측 API

이 API는 센서 데이터를 기반으로 작물의 생장 잠재력을 예측하는 서비스를 제공합니다.

## 모델 설명

### 1. 모델 아키텍처

-   **기반 모델**: LightGBM (Light Gradient Boosting Machine)
-   **학습 방식**: 그래디언트 부스팅 (Gradient Boosting)
-   **트리 구조**:
    -   최대 깊이: 6
    -   학습률: 0.1
    -   트리 개수: 100
    -   리프 노드 최소 샘플 수: 20

### 2. 입력 특성

모델은 다음과 같은 22개의 환경 및 생장 조건을 입력으로 받습니다:

#### 기본 환경 조건

-   **온도** (temperature): 작물 생장에 적합한 온도 범위 (°C)
-   **습도** (humidity): 대기 중 수분 함량 (%)
-   **pH**: 토양의 산성도 (0-14)
-   **강수량** (rainfall): 일일 강수량 (mm)
-   **토양 수분** (soil_moisture): 토양의 수분 함량 (%)
-   **일조량** (sunlight_exposure): 일일 일조 시간 (시간)

#### 영양소 조건

-   **질소** (N): 토양 내 질소 함량 (0-140)
-   **인** (P): 토양 내 인 함량 (0-140)
-   **칼륨** (K): 토양 내 칼륨 함량 (0-140)

#### 토양 조건

-   **토양 유형** (soil_type): 1(사질토), 2(양토), 3(점토)
-   **유기물 함량** (organic_matter): 토양 내 유기물 비율 (%)
-   **관개 빈도** (irrigation_frequency): 주간 관개 횟수
-   **수원 유형** (water_source_type): 1(지하수), 2(강수), 3(관개수)

#### 스트레스 조건

-   **풍속** (wind_speed): 평균 풍속 (km/h)
-   **CO2 농도** (co2_concentration): 대기 중 CO2 농도 (ppm)
-   **작물 밀도** (crop_density): 단위 면적당 작물 수 (식물/m²)
-   **해충 압력** (pest_pressure): 해충 피해 정도 (0-100)
-   **도시 지역 근접도** (urban_area_proximity): 도시와의 거리 (km)
-   **서리 위험도** (frost_risk): 서리 발생 가능성 (0-100)

#### 관리 조건

-   **비료 사용량** (fertilizer_usage): 단위 면적당 비료 사용량 (kg/ha)
-   **생장 단계** (growth_stage): 1(초기), 2(중기), 3(후기)
-   **수분 이용 효율** (water_usage_efficiency): 수분 흡수 효율성 (0-5)

### 3. 출력

-   **생장 잠재력 점수**: 0-100 사이의 정규화된 점수
-   **평가 등급**:
    -   80-100: 매우 좋음
    -   60-79: 좋음
    -   40-59: 보통
    -   20-39: 나쁨
    -   0-19: 매우 나쁨

### 4. 모델 성능

-   **평가 지표**:
    -   RMSE (Root Mean Square Error): 0.15
    -   R² Score: 0.85
    -   MAE (Mean Absolute Error): 0.12

### 5. 데이터 전처리

-   **정규화**: Min-Max 스케일링 (0-1 범위)
-   **범주형 변수**: 원-핫 인코딩
-   **결측치 처리**: 중앙값 대체

### 6. 모델 한계

-   특정 작물 종에 최적화되어 있음
-   극단적인 기상 조건에서는 예측 정확도가 감소할 수 있음
-   새로운 작물 종에 대해서는 재학습이 필요할 수 있음

## 설치 방법

1. 가상환경 생성 및 활성화:

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
.\venv\Scripts\activate  # Windows
```

2. 필요한 패키지 설치:

```bash
pip install -r requirements.txt
```

## 실행 방법

```bash
uvicorn main:app --reload
```

서버가 실행되면 다음 URL에서 API 문서를 확인할 수 있습니다:

-   Swagger UI: http://localhost:8000/docs
-   ReDoc: http://localhost:8000/redoc

## API 엔드포인트

### 1. 생장 잠재력 예측

-   **URL**: `/predict`
-   **Method**: POST
-   **Request Body**:

```json
{
    "N": 70.0,
    "P": 45.0,
    "K": 35.0,
    "temperature": 22.0,
    "humidity": 70.0,
    "ph": 6.5,
    "rainfall": 180.0,
    "soil_moisture": 20.0,
    "soil_type": 2,
    "sunlight_exposure": 8.0,
    "wind_speed": 10.0,
    "co2_concentration": 400.0,
    "organic_matter": 5.0,
    "irrigation_frequency": 3,
    "crop_density": 12.0,
    "pest_pressure": 50.0,
    "fertilizer_usage": 125.0,
    "growth_stage": 2,
    "urban_area_proximity": 25.0,
    "water_source_type": 2,
    "frost_risk": 50.0,
    "water_usage_efficiency": 3.0
}
```

-   **Response**:

```json
{
    "growth_potential": 85.5,
    "evaluation": "매우 좋음"
}
```

### 2. 서버 상태 확인

-   **URL**: `/health`
-   **Method**: GET
-   **Response**:

```json
{
    "status": "healthy"
}
```

## 모델 파일 위치

예측 모델 파일은 `models/crop_growth_model.pkl`에 위치해야 합니다.
