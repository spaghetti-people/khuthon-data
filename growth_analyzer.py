import sqlite3
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime, date


class GrowthAnalyzer:
    def __init__(self, db_path: str = "crop_conditions.db"):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

    def analyze_growth_condition(
        self, crop_name: str, current_conditions: Dict, planting_date: date = None
    ) -> Dict:
        """
        현재 작물의 상태를 분석하여 성장 단계와 상태를 반환합니다.

        Args:
            crop_name (str): 작물 이름
            current_conditions (Dict): 현재 환경 조건
                {
                    'temperature': float,
                    'humidity': float,
                    'ph': float,
                    'rainfall': float,
                    'soil_moisture': float,
                    'sunlight_exposure': float,
                    'water_usage_efficiency': float,
                    'N': float,
                    'P': float,
                    'K': float,
                    'soil_type': float,
                    'wind_speed': float,
                    'co2_concentration': float,
                    'crop_density': float,
                    'pest_pressure': float,
                    'urban_area_proximity': float,
                    'frost_risk': float
                }
            planting_date (date): 파종일 (선택사항)

        Returns:
            Dict: 분석 결과
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
        result = self.cursor.fetchone()
        if not result:
            raise ValueError(f"작물 '{crop_name}'을(를) 찾을 수 없습니다.")
        crop_id, total_growth_days = result

        # 현재 날짜 기준 성장 일수 계산
        current_date = date.today()
        if planting_date:
            days_since_planting = (current_date - planting_date).days
            growth_progress = min(100, (days_since_planting / total_growth_days) * 100)
        else:
            days_since_planting = None
            growth_progress = None

        # 각 성장 단계별 조건 가져오기
        stages = ["initial", "mid", "late"]
        stage_conditions = {}

        for stage in stages:
            self.cursor.execute(
                """
                SELECT stage_id, start_day, end_day 
                FROM growth_stages 
                WHERE stage_name = ?
            """,
                (stage,),
            )
            stage_info = self.cursor.fetchone()
            stage_id, start_day, end_day = stage_info

            # 기본 환경 조건
            self.cursor.execute(
                """
                SELECT * FROM basic_conditions 
                WHERE crop_id = ? AND stage_id = ?
            """,
                (crop_id, stage_id),
            )
            basic = self.cursor.fetchone()

            # 영양분 조건
            self.cursor.execute(
                """
                SELECT * FROM nutrient_conditions 
                WHERE crop_id = ? AND stage_id = ?
            """,
                (crop_id, stage_id),
            )
            nutrient = self.cursor.fetchone()

            # 토양 조건
            self.cursor.execute(
                """
                SELECT * FROM soil_conditions 
                WHERE crop_id = ? AND stage_id = ?
            """,
                (crop_id, stage_id),
            )
            soil = self.cursor.fetchone()

            # 스트레스 조건
            self.cursor.execute(
                """
                SELECT * FROM stress_conditions 
                WHERE crop_id = ? AND stage_id = ?
            """,
                (crop_id, stage_id),
            )
            stress = self.cursor.fetchone()

            stage_conditions[stage] = {
                "basic": basic,
                "nutrient": nutrient,
                "soil": soil,
                "stress": stress,
                "start_day": start_day,
                "end_day": end_day,
            }

        # 현재 상태와 각 단계별 조건 비교
        stage_scores = {}
        for stage in stages:
            score = self._calculate_stage_score(
                current_conditions, stage_conditions[stage]
            )

            # 날짜 정보가 있는 경우 해당 단계의 기간에 맞는지 확인
            if days_since_planting is not None:
                start_day = stage_conditions[stage]["start_day"]
                end_day = stage_conditions[stage]["end_day"]
                if start_day <= days_since_planting <= end_day:
                    score *= 1.2  # 해당 기간에 맞는 단계면 점수 가중치 부여
                else:
                    score *= 0.8  # 해당 기간이 아닌 단계면 점수 감소

            stage_scores[stage] = score

        # 가장 적합한 성장 단계 결정
        best_stage = max(stage_scores.items(), key=lambda x: x[1])

        # 개선이 필요한 조건 분석
        improvements = self._analyze_improvements(
            current_conditions, stage_conditions[best_stage[0]]
        )

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
                    stage_conditions[best_stage[0]]["basic"][1],  # stage_id
                    growth_progress,
                ),
            )
            self.conn.commit()

        return {
            "current_stage": best_stage[0],
            "stage_score": best_stage[1],
            "stage_scores": stage_scores,
            "improvements_needed": improvements,
            "days_since_planting": days_since_planting,
            "growth_progress": growth_progress,
            "total_growth_days": total_growth_days,
        }

    def _calculate_stage_score(self, current: Dict, stage_conditions: Dict) -> float:
        """현재 상태가 특정 성장 단계의 조건과 얼마나 일치하는지 점수를 계산합니다."""
        score = 0.0
        total_conditions = 0

        # 기본 환경 조건 점수 계산
        if stage_conditions["basic"]:
            basic_conditions = [
                ("temperature", "temperature"),
                ("humidity", "humidity"),
                ("ph", "ph"),
                ("rainfall", "rainfall"),
                ("soil_moisture", "soil_moisture"),
                ("sunlight_exposure", "sunlight_exposure"),
                ("water_usage_efficiency", "water_usage_efficiency"),
            ]

            for current_key, db_key in basic_conditions:
                if (
                    current_key in current
                    and f"{db_key}_min" in stage_conditions["basic"]
                ):
                    min_val = stage_conditions["basic"][f"{db_key}_min"]
                    max_val = stage_conditions["basic"][f"{db_key}_max"]
                    current_val = current[current_key]

                    if min_val <= current_val <= max_val:
                        score += 1.0
                    total_conditions += 1

        # 영양분 조건 점수 계산
        if stage_conditions["nutrient"]:
            nutrient_conditions = [
                ("N", "nitrogen"),
                ("P", "phosphorus"),
                ("K", "potassium"),
            ]

            for current_key, db_key in nutrient_conditions:
                if (
                    current_key in current
                    and f"{db_key}_min" in stage_conditions["nutrient"]
                ):
                    min_val = stage_conditions["nutrient"][f"{db_key}_min"]
                    max_val = stage_conditions["nutrient"][f"{db_key}_max"]
                    current_val = current[current_key]

                    if min_val <= current_val <= max_val:
                        score += 1.0
                    total_conditions += 1

        return (score / total_conditions) * 100 if total_conditions > 0 else 0

    def _analyze_improvements(
        self, current: Dict, stage_conditions: Dict
    ) -> List[Dict]:
        """개선이 필요한 조건들을 분석합니다."""
        improvements = []

        # 기본 환경 조건 분석
        if stage_conditions["basic"]:
            basic_conditions = [
                ("temperature", "temperature", "온도"),
                ("humidity", "humidity", "습도"),
                ("ph", "ph", "pH"),
                ("rainfall", "rainfall", "강수량"),
                ("soil_moisture", "soil_moisture", "토양 수분"),
                ("sunlight_exposure", "sunlight_exposure", "일조량"),
                ("water_usage_efficiency", "water_usage_efficiency", "물 사용 효율"),
            ]

            for current_key, db_key, name in basic_conditions:
                if (
                    current_key in current
                    and f"{db_key}_min" in stage_conditions["basic"]
                ):
                    min_val = stage_conditions["basic"][f"{db_key}_min"]
                    max_val = stage_conditions["basic"][f"{db_key}_max"]
                    current_val = current[current_key]

                    if current_val < min_val:
                        improvements.append(
                            {
                                "condition": name,
                                "current": current_val,
                                "recommended": f"{min_val} ~ {max_val}",
                                "action": f"{name}을(를) 높여주세요.",
                            }
                        )
                    elif current_val > max_val:
                        improvements.append(
                            {
                                "condition": name,
                                "current": current_val,
                                "recommended": f"{min_val} ~ {max_val}",
                                "action": f"{name}을(를) 낮춰주세요.",
                            }
                        )

        return improvements

    def close(self):
        """데이터베이스 연결을 종료합니다."""
        self.conn.close()


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
    planting_date = date.today().replace(day=date.today().day - 30)

    try:
        result = analyzer.analyze_growth_condition(
            "rice", current_conditions, planting_date
        )
        print(f"\n현재 성장 단계: {result['current_stage']}")
        print(f"단계 적합도 점수: {result['stage_score']:.2f}%")
        print(f"파종 후 경과일: {result['days_since_planting']}일")
        print(f"성장 진행률: {result['growth_progress']:.2f}%")

        print("\n각 단계별 점수:")
        for stage, score in result["stage_scores"].items():
            print(f"- {stage}: {score:.2f}%")

        if result["improvements_needed"]:
            print("\n개선이 필요한 조건:")
            for imp in result["improvements_needed"]:
                print(
                    f"- {imp['condition']}: 현재 {imp['current']}, 권장 {imp['recommended']}"
                )
                print(f"  조치: {imp['action']}")
        else:
            print("\n모든 조건이 적정 범위 내에 있습니다.")

    except ValueError as e:
        print(f"오류: {str(e)}")
    finally:
        analyzer.close()
