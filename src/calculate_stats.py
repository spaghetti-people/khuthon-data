# 실제 평균과 표준편차를 계산하는 함수
import pandas as pd
import numpy as np

# 데이터 로드
df = pd.read_csv("Crop_recommendationV2.csv")

# 수치형 특성 선택
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

# 평균과 표준편차 계산
means = df[numeric_features].mean()
stds = df[numeric_features].std()

print("\n=== 평균 ===")
for feature in numeric_features:
    print(f'"{feature}": {means[feature]:.2f},')

print("\n=== 표준편차 ===")
for feature in numeric_features:
    print(f'"{feature}": {stds[feature]:.2f},')
