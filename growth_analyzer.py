import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, date, timedelta


class GrowthAnalyzer:
    def __init__(self, db_path: str = "crop_conditions.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()

        # 성장 단계별 특징 패턴 정의
        self.stage_patterns = {
            "germination": {
                "soil_moisture": {"min": 0.7, "max": 0.9},
                "temperature": {"min": 20, "max": 25},
                "humidity": {"min": 0.8, "max": 0.9},
            },
            "seedling": {
                "soil_moisture": {"min": 0.6, "max": 0.8},
                "temperature": {"min": 22, "max": 28},
                "humidity": {"min": 0.7, "max": 0.85},
            },
            "vegetative": {
                "soil_moisture": {"min": 0.5, "max": 0.7},
                "temperature": {"min": 24, "max": 30},
                "humidity": {"min": 0.6, "max": 0.8},
            },
            "flowering": {
                "soil_moisture": {"min": 0.5, "max": 0.7},
                "temperature": {"min": 25, "max": 32},
                "humidity": {"min": 0.5, "max": 0.7},
            },
            "maturity": {
                "soil_moisture": {"min": 0.4, "max": 0.6},
                "temperature": {"min": 23, "max": 30},
                "humidity": {"min": 0.5, "max": 0.7},
            },
        }

    def record_sensor_data(self, crop_id: int, sensor_data: Dict) -> None:
        """
        센서 데이터를 기록합니다.
        """
        query = """
        INSERT INTO sensor_data (
            crop_id, temperature, humidity, ph, rainfall,
            soil_moisture, sunlight_exposure, co2_concentration,
            nitrogen, phosphorus, potassium,
            irrigation_frequency, fertilizer_usage
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        self.cursor.execute(
            query,
            (
                crop_id,
                sensor_data.get("temperature"),
                sensor_data.get("humidity"),
                sensor_data.get("ph"),
                sensor_data.get("rainfall"),
                sensor_data.get("soil_moisture"),
                sensor_data.get("sunlight_exposure"),
                sensor_data.get("co2_concentration"),
                sensor_data.get("nitrogen"),
                sensor_data.get("phosphorus"),
                sensor_data.get("potassium"),
                sensor_data.get("irrigation_frequency"),
                sensor_data.get("fertilizer_usage"),
            ),
        )
        self.conn.commit()

    def determine_growth_stage(
        self, crop_id: int, sensor_data: Dict, days_since_planting: int
    ) -> str:
        """
        센서 데이터와 경과일을 기반으로 성장 단계를 판단합니다.
        """
        # 기본적인 시간 기반 단계 확인
        self.cursor.execute(
            """
            SELECT stage_name, min_days, max_days
            FROM growth_stages
            WHERE ? BETWEEN min_days AND max_days
            """,
            (days_since_planting,),
        )
        time_based_stage = self.cursor.fetchone()

        if not time_based_stage:
            return "unknown"

        stage_name = time_based_stage["stage_name"]
        stage_pattern = self.stage_patterns[stage_name]

        # 센서 데이터 기반 패턴 매칭
        pattern_match_score = self._calculate_pattern_match_score(
            sensor_data, stage_pattern
        )

        # 패턴 매칭 점수가 낮으면 이전/다음 단계 확인
        if pattern_match_score < 0.6:
            if days_since_planting > time_based_stage["min_days"]:
                # 이전 단계 확인
                self.cursor.execute(
                    """
                    SELECT stage_name, min_days, max_days
                    FROM growth_stages
                    WHERE max_days < ?
                    ORDER BY max_days DESC
                    LIMIT 1
                    """,
                    (days_since_planting,),
                )
                prev_stage = self.cursor.fetchone()
                if prev_stage:
                    prev_pattern = self.stage_patterns[prev_stage["stage_name"]]
                    prev_score = self._calculate_pattern_match_score(
                        sensor_data, prev_pattern
                    )
                    if prev_score > pattern_match_score:
                        return prev_stage["stage_name"]

            if days_since_planting < time_based_stage["max_days"]:
                # 다음 단계 확인
                self.cursor.execute(
                    """
                    SELECT stage_name, min_days, max_days
                    FROM growth_stages
                    WHERE min_days > ?
                    ORDER BY min_days ASC
                    LIMIT 1
                    """,
                    (days_since_planting,),
                )
                next_stage = self.cursor.fetchone()
                if next_stage:
                    next_pattern = self.stage_patterns[next_stage["stage_name"]]
                    next_score = self._calculate_pattern_match_score(
                        sensor_data, next_pattern
                    )
                    if next_score > pattern_match_score:
                        return next_stage["stage_name"]

        return stage_name

    def _calculate_pattern_match_score(
        self, sensor_data: Dict, stage_pattern: Dict
    ) -> float:
        """
        센서 데이터와 단계 패턴의 일치도를 계산합니다.
        """
        scores = []

        for condition, pattern in stage_pattern.items():
            if condition in sensor_data:
                value = sensor_data[condition]
                min_val = pattern["min"]
                max_val = pattern["max"]

                if min_val <= value <= max_val:
                    scores.append(1.0)
                else:
                    # 범위를 벗어난 정도에 따라 점수 감소
                    if value < min_val:
                        scores.append(max(0, 1 - (min_val - value) / min_val))
                    else:
                        scores.append(max(0, 1 - (value - max_val) / max_val))

        return np.mean(scores) if scores else 0.0

    def analyze_growth(
        self,
        crop_name: str,
        current_conditions: Dict,
        planting_date: date = None,
        current_date: date = None,
    ) -> Dict:
        """
        작물의 현재 성장 상태를 분석합니다.
        """
        # 작물 ID 조회
        self.cursor.execute(
            "SELECT crop_id FROM crops WHERE crop_name = ?", (crop_name,)
        )
        crop = self.cursor.fetchone()
        if not crop:
            raise ValueError(f"작물 '{crop_name}'을(를) 찾을 수 없습니다.")
        crop_id = crop["crop_id"]

        # 현재 날짜 기준 성장 일수 계산
        if planting_date:
            if current_date is None:
                current_date = date.today()
            days_since_planting = (current_date - planting_date).days
            if days_since_planting < 0:
                raise ValueError("파종일이 현재 날짜보다 미래입니다.")
        else:
            days_since_planting = 0

        # 센서 데이터 기록
        self.record_sensor_data(crop_id, current_conditions)

        # 현재 성장 단계 결정
        current_stage = self.determine_growth_stage(
            crop_id, current_conditions, days_since_planting
        )

        # 현재 단계의 이상적인 조건 조회
        ideal_conditions = self.get_ideal_conditions(crop_id, current_stage)

        # 현재 조건 분석
        condition_analysis = self.analyze_conditions(
            current_conditions, ideal_conditions
        )

        # 성장 진행도 계산
        growth_progress = self.calculate_growth_progress(
            days_since_planting, ideal_conditions["total_days"], condition_analysis
        )

        # 남은 성장 진행도 계산
        remaining_progress = max(0, 100 - growth_progress)

        # 성장 기록 저장
        self.cursor.execute(
            """
            INSERT INTO growth_records (
                crop_id, stage_id, growth_progress, condition_score
            ) VALUES (
                ?, 
                (SELECT stage_id FROM growth_stages WHERE stage_name = ?),
                ?,
                ?
            )
            """,
            (
                crop_id,
                current_stage,
                growth_progress,
                (
                    sum(condition_analysis.values()) / len(condition_analysis)
                    if condition_analysis
                    else 0
                ),
            ),
        )
        self.conn.commit()

        return {
            "current_stage": current_stage,
            "growth_progress": growth_progress,
            "remaining_progress": remaining_progress,
            "days_since_planting": days_since_planting,
            "ideal_conditions": ideal_conditions,
            "condition_analysis": condition_analysis,
        }

    def calculate_growth_progress(
        self, days_since_planting: int, total_growth_days: int, condition_analysis: Dict
    ) -> float:
        """
        성장 진행도를 계산합니다.

        Args:
            days_since_planting (int): 파종 후 경과일수
            total_growth_days (int): 총 성장 기간
            condition_analysis (Dict): 조건 분석 결과

        Returns:
            float: 성장 진행도 (%)
        """
        # 기본 진행도 (시간 기반)
        base_progress = min((days_since_planting / total_growth_days) * 100, 100.0)

        # 조건별 가중치 계산
        condition_weights = {
            "temperature": 0.2,
            "humidity": 0.15,
            "ph": 0.1,
            "rainfall": 0.1,
            "soil_moisture": 0.15,
            "sunlight_exposure": 0.15,
            "nutrients": 0.15,
        }

        # 조건별 진행도 조정
        condition_adjustment = 0
        for condition, weight in condition_weights.items():
            if condition in condition_analysis:
                condition_adjustment += condition_analysis[condition] * weight

        # 최종 진행도 계산 (기본 진행도 * 조건 조정)
        final_progress = base_progress * condition_adjustment

        # 진행도가 100%를 넘지 않도록 제한
        return min(final_progress, 100.0)

    def get_growth_status_message(self, analysis_result: Dict) -> str:
        """
        성장 상태 메시지를 생성합니다.

        Args:
            analysis_result (Dict): 성장 분석 결과

        Returns:
            str: 성장 상태 메시지
        """
        days = analysis_result["days_since_planting"]
        progress = analysis_result["growth_progress"]
        remaining = analysis_result["remaining_progress"]

        return (
            f"이 환경에서 {days}일 동안 자랐다면, "
            f"이상적인 조건에서의 성장 대비 {progress:.1f}% 성장했을 것으로 예상됩니다.\n"
            f"앞으로 성장까지 {remaining:.1f}% 남았습니다."
        )

    def close(self):
        """데이터베이스 연결을 종료합니다."""
        self.conn.close()

    def get_ideal_conditions(self, crop_id: int, stage: str) -> Dict:
        """
        현재 성장 단계의 이상적인 조건을 조회합니다.
        """
        # 성장 단계 ID 조회
        self.cursor.execute(
            "SELECT stage_id FROM growth_stages WHERE stage_name = ?", (stage,)
        )
        stage_result = self.cursor.fetchone()
        if not stage_result:
            return {
                "total_days": 100,  # 기본 성장 기간
                "temperature": {"min": 20, "max": 30},
                "humidity": {"min": 60, "max": 80},
                "ph": {"min": 5.5, "max": 7.0},
                "rainfall": {"min": 80, "max": 120},
                "soil_moisture": {"min": 60, "max": 80},
                "sunlight_exposure": {"min": 6, "max": 12},
                "co2_concentration": {"min": 350, "max": 450},
                "nitrogen": {"min": 20, "max": 40},
                "phosphorus": {"min": 20, "max": 40},
                "potassium": {"min": 20, "max": 40},
                "irrigation_frequency": {"min": 1, "max": 3},
                "fertilizer_usage": {"min": 1.0, "max": 2.0},
            }

        stage_id = stage_result["stage_id"]

        # 기본 조건 조회
        self.cursor.execute(
            """
            SELECT min_temperature, max_temperature,
                   min_humidity, max_humidity,
                   min_ph, max_ph,
                   min_rainfall, max_rainfall,
                   min_co2_concentration, max_co2_concentration
            FROM basic_conditions
            WHERE crop_id = ? AND stage_id = ?
            """,
            (crop_id, stage_id),
        )
        basic_result = self.cursor.fetchone()

        # 영양소 조건 조회
        self.cursor.execute(
            """
            SELECT min_nitrogen, max_nitrogen,
                   min_phosphorus, max_phosphorus,
                   min_potassium, max_potassium
            FROM nutrient_conditions
            WHERE crop_id = ? AND stage_id = ?
            """,
            (crop_id, stage_id),
        )
        nutrient_result = self.cursor.fetchone()

        # 토양 조건 조회
        self.cursor.execute(
            """
            SELECT min_soil_moisture, max_soil_moisture
            FROM soil_conditions
            WHERE crop_id = ? AND stage_id = ?
            """,
            (crop_id, stage_id),
        )
        soil_result = self.cursor.fetchone()

        # 관리 조건 조회
        self.cursor.execute(
            """
            SELECT irrigation_frequency, fertilizer_usage
            FROM management_conditions
            WHERE crop_id = ? AND stage_id = ?
            """,
            (crop_id, stage_id),
        )
        management_result = self.cursor.fetchone()

        # 데이터베이스에 조건이 없는 경우 기본값 사용
        if not all([basic_result, nutrient_result, soil_result, management_result]):
            return {
                "total_days": 100,  # 기본 성장 기간
                "temperature": {"min": 20, "max": 30},
                "humidity": {"min": 60, "max": 80},
                "ph": {"min": 5.5, "max": 7.0},
                "rainfall": {"min": 80, "max": 120},
                "soil_moisture": {"min": 60, "max": 80},
                "sunlight_exposure": {"min": 6, "max": 12},
                "co2_concentration": {"min": 350, "max": 450},
                "nitrogen": {"min": 20, "max": 40},
                "phosphorus": {"min": 20, "max": 40},
                "potassium": {"min": 20, "max": 40},
                "irrigation_frequency": {"min": 1, "max": 3},
                "fertilizer_usage": {"min": 1.0, "max": 2.0},
            }

        # 총 성장 기간 조회
        self.cursor.execute(
            "SELECT total_growth_days FROM crops WHERE crop_id = ?", (crop_id,)
        )
        total_days = self.cursor.fetchone()["total_growth_days"]

        return {
            "total_days": total_days,
            "temperature": {
                "min": basic_result["min_temperature"],
                "max": basic_result["max_temperature"],
            },
            "humidity": {
                "min": basic_result["min_humidity"],
                "max": basic_result["max_humidity"],
            },
            "ph": {"min": basic_result["min_ph"], "max": basic_result["max_ph"]},
            "rainfall": {
                "min": basic_result["min_rainfall"],
                "max": basic_result["max_rainfall"],
            },
            "soil_moisture": {
                "min": soil_result["min_soil_moisture"],
                "max": soil_result["max_soil_moisture"],
            },
            "co2_concentration": {
                "min": basic_result["min_co2_concentration"],
                "max": basic_result["max_co2_concentration"],
            },
            "nitrogen": {
                "min": nutrient_result["min_nitrogen"],
                "max": nutrient_result["max_nitrogen"],
            },
            "phosphorus": {
                "min": nutrient_result["min_phosphorus"],
                "max": nutrient_result["max_phosphorus"],
            },
            "potassium": {
                "min": nutrient_result["min_potassium"],
                "max": nutrient_result["max_potassium"],
            },
            "irrigation_frequency": {
                "min": management_result["irrigation_frequency"] - 1,
                "max": management_result["irrigation_frequency"] + 1,
            },
            "fertilizer_usage": {
                "min": management_result["fertilizer_usage"] * 0.8,
                "max": management_result["fertilizer_usage"] * 1.2,
            },
        }

    def analyze_conditions(
        self, current_conditions: Dict, ideal_conditions: Dict
    ) -> Dict:
        """
        현재 조건과 이상적인 조건을 비교하여 점수를 계산합니다.
        """
        scores = {}

        for condition, value in current_conditions.items():
            if condition in ideal_conditions:
                ideal = ideal_conditions[condition]
                min_val = ideal["min"]
                max_val = ideal["max"]

                if min_val <= value <= max_val:
                    scores[condition] = 1.0
                else:
                    # 범위를 벗어난 정도에 따라 점수 감소
                    if value < min_val:
                        scores[condition] = max(0, 1 - (min_val - value) / min_val)
                    else:
                        scores[condition] = max(0, 1 - (value - max_val) / max_val)

        return scores


# 사용 예시
if __name__ == "__main__":
    analyzer = GrowthAnalyzer()

    # 현재 상태 예시
    current_conditions = {
        "temperature": 25.0,
        "humidity": 80.0,
        "ph": 6.0,
        "rainfall": 100.0,
        "soil_moisture": 20.0,
        "sunlight_exposure": 8.0,
        "water_usage_efficiency": 3.0,
        "N": 30.0,
        "P": 40.0,
        "K": 30.0,
        "soil_type": 2.0,
        "wind_speed": 10.0,
        "co2_concentration": 400.0,
        "crop_density": 10.0,
        "pest_pressure": 50.0,
        "urban_area_proximity": 20.0,
        "frost_risk": 30.0,
    }

    # 파종일 예시 (30일 전)
    planting_date = date.today() - timedelta(days=30)

    try:
        result = analyzer.analyze_growth("rice", current_conditions, planting_date)
        print(f"\n현재 성장 단계: {result['current_stage']}")
        print(f"성장 진행도: {result['growth_progress']:.2f}%")
        print(f"남은 성장 진행도: {result['remaining_progress']:.2f}%")
        print(f"파종 후 경과일: {result['days_since_planting']}일")

        print("\n각 조건별 분석 결과:")
        for condition, analysis in result["condition_analysis"].items():
            print(f"- {condition}: {analysis}")

        print("\n성장 상태 메시지:")
        print(analyzer.get_growth_status_message(result))

    except ValueError as e:
        print(f"오류: {str(e)}")
    finally:
        analyzer.close()
