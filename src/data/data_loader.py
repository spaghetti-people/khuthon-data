import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CropDataLoader:
    def __init__(self, data_file):
        self.data_file = data_file
        self.df = None

    def load_data(self):
        """CSV 파일에서 데이터를 로드합니다."""
        try:
            self.df = pd.read_csv(self.data_file)
            logger.info(f"데이터 로드 완료: {len(self.df)} 행")
            return self.df
        except Exception as e:
            logger.error(f"데이터 로드 중 오류 발생: {str(e)}")
            raise

    def preprocess(self):
        """데이터 전처리를 수행합니다."""
        if self.df is None:
            raise ValueError("데이터가 로드되지 않았습니다.")

        # 결측치 확인
        missing_values = self.df.isnull().sum()
        if missing_values.any():
            logger.warning(f"결측치 발견:\n{missing_values[missing_values > 0]}")

        # 수치형 특성 Min-Max 정규화 (0-1 범위로)
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            min_val = self.df[col].min()
            max_val = self.df[col].max()
            self.df[col] = (self.df[col] - min_val) / (max_val - min_val)

        # 범주형 변수 원-핫 인코딩
        categorical_cols = ["soil_type", "water_source_type"]
        self.df = pd.get_dummies(self.df, columns=categorical_cols, drop_first=True)

        return self.df

    def create_growth_potential_score(self):
        """작물의 성장 잠재력을 계산합니다."""
        if self.df is None:
            raise ValueError("데이터가 로드되지 않았습니다.")

        # 성장 잠재력 점수 계산을 위한 가중치 설정
        weights = {
            "N": 0.1,
            "P": 0.1,
            "K": 0.1,
            "temperature": 0.1,
            "humidity": 0.1,
            "ph": 0.1,
            "rainfall": 0.1,
            "soil_moisture": 0.1,
            "sunlight_exposure": 0.1,
            "co2_concentration": 0.05,
            "organic_matter": 0.05,
            "irrigation_frequency": 0.05,
            "water_usage_efficiency": 0.05,
        }

        # 데이터에서 각 작물별 최적값 계산
        optimal_values = {}
        for crop in self.df["label"].unique():
            crop_data = self.df[self.df["label"] == crop]

            # 각 특성별 평균값을 최적값으로 설정
            optimal_values[crop] = {
                "N": crop_data["N"].mean(),
                "P": crop_data["P"].mean(),
                "K": crop_data["K"].mean(),
                "temperature": crop_data["temperature"].mean(),
                "humidity": crop_data["humidity"].mean(),
                "ph": crop_data["ph"].mean(),
                "rainfall": crop_data["rainfall"].mean(),
                "soil_moisture": crop_data["soil_moisture"].mean(),
                "sunlight_exposure": crop_data["sunlight_exposure"].mean(),
                "co2_concentration": crop_data["co2_concentration"].mean(),
                "organic_matter": crop_data["organic_matter"].mean(),
                "irrigation_frequency": crop_data["irrigation_frequency"].mean(),
                "water_usage_efficiency": crop_data["water_usage_efficiency"].mean(),
            }

        # 성장 잠재력 점수 계산
        growth_potential = np.zeros(len(self.df))
        for idx, row in self.df.iterrows():
            crop = row["label"]
            current_optimal_values = optimal_values[crop]

            for feature, weight in weights.items():
                if feature in self.df.columns:
                    # 현재 값과 최적값의 차이를 계산 (0-1 범위)
                    diff = 1 - abs(row[feature] - current_optimal_values[feature])
                    # 차이값을 0-1 범위로 정규화하고 가중치 적용
                    growth_potential[idx] += weight * diff

        # 점수를 0-1 범위로 정규화
        min_score = growth_potential.min()
        max_score = growth_potential.max()
        growth_potential = (growth_potential - min_score) / (max_score - min_score)

        self.df["growth_potential"] = growth_potential
        return self.df
