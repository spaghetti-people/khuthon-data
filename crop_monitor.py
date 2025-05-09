from growth_analyzer import GrowthAnalyzer
from typing import Dict


class CropMonitor:
    def __init__(self):
        self.analyzer = GrowthAnalyzer()

    def check_crop_health(self, crop_name: str, sensor_data: Dict) -> Dict:
        """
        작물의 건강 상태를 확인합니다.

        Args:
            crop_name (str): 작물 이름
            sensor_data (Dict): 센서에서 수집된 데이터

        Returns:
            Dict: 분석 결과
        """
        try:
            # 센서 데이터를 분석기에 맞는 형식으로 변환
            current_conditions = {
                "temperature": sensor_data.get("temp", 0),
                "humidity": sensor_data.get("humidity", 0),
                "ph": sensor_data.get("ph", 0),
                "rainfall": sensor_data.get("rain", 0),
                "soil_moisture": sensor_data.get("soil_moisture", 0),
                "sunlight_exposure": sensor_data.get("sunlight", 0),
                "water_usage_efficiency": sensor_data.get("water_efficiency", 0),
                "N": sensor_data.get("nitrogen", 0),
                "P": sensor_data.get("phosphorus", 0),
                "K": sensor_data.get("potassium", 0),
                "soil_type": sensor_data.get("soil_type", 0),
                "wind_speed": sensor_data.get("wind", 0),
                "co2_concentration": sensor_data.get("co2", 0),
                "crop_density": sensor_data.get("density", 0),
                "pest_pressure": sensor_data.get("pest", 0),
                "urban_area_proximity": sensor_data.get("urban", 0),
                "frost_risk": sensor_data.get("frost", 0),
            }

            # 성장 상태 분석
            analysis_result = self.analyzer.analyze_growth_condition(
                crop_name, current_conditions
            )

            # 결과 가공
            health_status = {
                "crop_name": crop_name,
                "growth_stage": analysis_result["current_stage"],
                "health_score": analysis_result["stage_score"],
                "stage_scores": analysis_result["stage_scores"],
                "recommendations": [],
            }

            # 개선 사항 추가
            for imp in analysis_result["improvements_needed"]:
                health_status["recommendations"].append(
                    {
                        "condition": imp["condition"],
                        "current_value": imp["current"],
                        "recommended_range": imp["recommended"],
                        "action": imp["action"],
                    }
                )

            return health_status

        except Exception as e:
            return {"error": str(e), "crop_name": crop_name}

    def close(self):
        """분석기 연결을 종료합니다."""
        self.analyzer.close()


# 사용 예시
if __name__ == "__main__":
    # 센서 데이터 예시
    sensor_data = {
        "temp": 25.0,
        "humidity": 80.0,
        "ph": 6.0,
        "rain": 100.0,
        "soil_moisture": 20.0,
        "sunlight": 8.0,
        "water_efficiency": 3.0,
        "nitrogen": 30.0,
        "phosphorus": 40.0,
        "potassium": 30.0,
        "soil_type": 2.0,
        "wind": 10.0,
        "co2": 400.0,
        "density": 10.0,
        "pest": 50.0,
        "urban": 20.0,
        "frost": 30.0,
    }

    monitor = CropMonitor()
    try:
        # 벼 작물의 건강 상태 확인
        result = monitor.check_crop_health("rice", sensor_data)

        if "error" in result:
            print(f"오류 발생: {result['error']}")
        else:
            print(f"\n작물: {result['crop_name']}")
            print(f"현재 성장 단계: {result['growth_stage']}")
            print(f"건강 점수: {result['health_score']:.2f}%")

            print("\n각 단계별 점수:")
            for stage, score in result["stage_scores"].items():
                print(f"- {stage}: {score:.2f}%")

            if result["recommendations"]:
                print("\n개선 권장사항:")
                for rec in result["recommendations"]:
                    print(f"- {rec['condition']}:")
                    print(f"  현재 값: {rec['current_value']}")
                    print(f"  권장 범위: {rec['recommended_range']}")
                    print(f"  조치: {rec['action']}")
            else:
                print("\n모든 조건이 적정 범위 내에 있습니다.")

    finally:
        monitor.close()
