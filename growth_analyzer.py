import sqlite3
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime, date, timedelta


class GrowthAnalyzer:
    def __init__(self, db_path: str = "crop_conditions.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()

    def analyze_growth(
        self, crop_name: str, current_conditions: Dict, planting_date: date = None
    ) -> Dict:
        """
        작물의 현재 성장 상태를 분석합니다.

        Args:
            crop_name (str): 작물 이름
            current_conditions (Dict): 현재 환경 조건
            planting_date (date): 파종일 (선택사항)

        Returns:
            Dict: {
                'current_stage': str,  # 현재 성장 단계
                'growth_progress': float,  # 현재 성장 진행도 (%)
                'remaining_progress': float,  # 남은 성장 진행도 (%)
                'days_since_planting': int,  # 파종 후 경과일수
                'ideal_conditions': Dict,  # 현재 단계의 이상적인 조건
                'condition_analysis': Dict  # 각 조건별 분석 결과
            }
        """
        # 작물 ID와 총 성장 기간 가져오기
        self.cursor.execute(
            """
            SELECT crop_id, total_growth_days 
            FROM crops 
            WHERE crop_name = ?
        """,
            (crop_name,),
        )
        crop = self.cursor.fetchone()
        if not crop:
            raise ValueError(f"작물 '{crop_name}'을(를) 찾을 수 없습니다.")
        crop_id, total_growth_days = crop

        # 현재 날짜 기준 성장 일수 계산
        current_date = date.today()
        if planting_date:
            days_since_planting = (current_date - planting_date).days
        else:
            days_since_planting = 0

        # 현재 성장 단계 결정
        current_stage = self.determine_growth_stage(crop_id, days_since_planting)

        # 현재 단계의 이상적인 조건 조회
        ideal_conditions = self.get_ideal_conditions(crop_id, current_stage)

        # 현재 조건 분석
        condition_analysis = self.analyze_conditions(
            current_conditions, ideal_conditions
        )

        # 성장 진행도 계산
        growth_progress = self.calculate_growth_progress(
            days_since_planting, total_growth_days, condition_analysis
        )

        # 남은 성장 진행도 계산
        remaining_progress = 100 - growth_progress

        # 성장 기록 저장
        if planting_date:
            self.cursor.execute(
                """
                INSERT INTO growth_records 
                (crop_id, planting_date, current_date, current_stage_id, growth_progress)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    crop_id,
                    planting_date,
                    current_date,
                    current_stage,
                    growth_progress,
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
        base_progress = (days_since_planting / total_growth_days) * 100

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

    def determine_growth_stage(self, crop_id: int, days_since_planting: int) -> str:
        """
        현재 성장 단계를 결정합니다.

        Args:
            crop_id (int): 작물 ID
            days_since_planting (int): 파종 후 경과일수

        Returns:
            str: 현재 성장 단계 ('initial', 'mid', 'late' 중 하나)
        """
        self.cursor.execute(
            """
            SELECT total_growth_days 
            FROM crops 
            WHERE crop_id = ?
        """,
            (crop_id,),
        )
        total_days = self.cursor.fetchone()["total_growth_days"]

        # 성장 단계별 기간 비율 (초기: 20%, 중기: 50%, 후기: 30%)
        initial_end = total_days * 0.2
        mid_end = total_days * 0.7

        if days_since_planting <= initial_end:
            return "initial"
        elif days_since_planting <= mid_end:
            return "mid"
        else:
            return "late"

    def get_ideal_conditions(self, crop_id: int, stage: str) -> Dict:
        """
        현재 단계의 이상적인 조건을 가져옵니다.

        Args:
            crop_id (int): 작물 ID
            stage (str): 성장 단계

        Returns:
            Dict: 이상적인 조건들
        """
        # 성장 단계 ID 조회
        self.cursor.execute(
            "SELECT stage_id FROM growth_stages WHERE stage_name = ?", (stage,)
        )
        stage_id = self.cursor.fetchone()["stage_id"]

        # 기본 조건
        self.cursor.execute(
            """
            SELECT * FROM basic_conditions 
            WHERE crop_id = ? AND stage_id = ?
        """,
            (crop_id, stage_id),
        )
        basic = dict(self.cursor.fetchone())

        # 영양분 조건
        self.cursor.execute(
            """
            SELECT * FROM nutrient_conditions 
            WHERE crop_id = ? AND stage_id = ?
        """,
            (crop_id, stage_id),
        )
        nutrient = dict(self.cursor.fetchone())

        # 토양 조건
        self.cursor.execute(
            """
            SELECT * FROM soil_conditions 
            WHERE crop_id = ? AND stage_id = ?
        """,
            (crop_id, stage_id),
        )
        soil = dict(self.cursor.fetchone())

        # 스트레스 조건
        self.cursor.execute(
            """
            SELECT * FROM stress_conditions 
            WHERE crop_id = ? AND stage_id = ?
        """,
            (crop_id, stage_id),
        )
        stress = dict(self.cursor.fetchone())

        return {
            "basic": basic,
            "nutrient": nutrient,
            "soil": soil,
            "stress": stress,
        }

    def analyze_conditions(self, current: Dict, ideal: Dict) -> Dict:
        """
        현재 조건을 이상적인 조건과 비교하여 분석합니다.

        Args:
            current (Dict): 현재 조건
            ideal (Dict): 이상적인 조건

        Returns:
            Dict: 각 조건별 적합도 (0~1)
        """
        analysis = {}

        # 기본 조건 분석
        basic_conditions = [
            ("temperature", "temperature"),
            ("humidity", "humidity"),
            ("ph", "ph"),
            ("rainfall", "rainfall"),
            ("soil_moisture", "soil_moisture"),
            ("sunlight_exposure", "sunlight_exposure"),
        ]

        for current_key, db_key in basic_conditions:
            if current_key in current:
                min_val = ideal["basic"][f"{db_key}_min"]
                max_val = ideal["basic"][f"{db_key}_max"]
                current_val = current[current_key]

                if min_val <= current_val <= max_val:
                    analysis[current_key] = 1.0
                else:
                    # 범위를 벗어난 정도에 따라 점수 감소
                    if current_val < min_val:
                        analysis[current_key] = max(
                            0, 1 - (min_val - current_val) / min_val
                        )
                    else:
                        analysis[current_key] = max(
                            0, 1 - (current_val - max_val) / max_val
                        )

        # 영양분 조건 분석
        nutrient_conditions = [
            ("N", "nitrogen"),
            ("P", "phosphorus"),
            ("K", "potassium"),
        ]

        nutrient_score = 0
        for current_key, db_key in nutrient_conditions:
            if current_key in current:
                min_val = ideal["nutrient"][f"{db_key}_min"]
                max_val = ideal["nutrient"][f"{db_key}_max"]
                current_val = current[current_key]

                if min_val <= current_val <= max_val:
                    nutrient_score += 1
                else:
                    # 범위를 벗어난 정도에 따라 점수 감소
                    if current_val < min_val:
                        nutrient_score += max(0, 1 - (min_val - current_val) / min_val)
                    else:
                        nutrient_score += max(0, 1 - (current_val - max_val) / max_val)

        analysis["nutrients"] = nutrient_score / len(nutrient_conditions)

        return analysis


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
