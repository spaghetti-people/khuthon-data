from growth_analyzer import GrowthAnalyzer
from datetime import datetime, date, timedelta


def test_growth_analyzer():
    # 분석기 초기화
    analyzer = GrowthAnalyzer()

    # 테스트할 작물과 날짜 설정
    crop_name = "rice"
    planting_date = date.today() - timedelta(days=35)

    # 현재 센서 데이터 (예시)
    current_conditions = {
        "temperature": 24.5,
        "humidity": 82.0,
        "ph": 6.0,
        "rainfall": 120.0,
        "soil_moisture": 25.0,
        "sunlight_exposure": 9.0,
        "water_usage_efficiency": 3.0,
        "N": 35.0,
        "P": 45.0,
        "K": 35.0,
        "soil_type": 2.0,
        "wind_speed": 8.0,
        "co2_concentration": 410.0,
        "crop_density": 12.0,
        "pest_pressure": 45.0,
        "urban_area_proximity": 25.0,
        "frost_risk": 20.0,
    }

    # 성장 상태 분석
    result = analyzer.analyze_growth(
        crop_name=crop_name,
        current_conditions=current_conditions,
        planting_date=planting_date,
    )

    # 결과 출력
    print("\n=== 작물 성장 상태 분석 결과 ===")
    print(f"작물: {crop_name}")
    print(f"파종일: {planting_date.strftime('%Y-%m-%d')}")
    print(f"현재 성장 단계: {result['current_stage']}")
    print(f"성장 진행도: {result['growth_progress']:.1f}%")
    print(f"남은 성장도: {result['remaining_progress']:.1f}%")
    print(f"파종 후 경과일: {result['days_since_planting']}일")

    print("\n=== 조건별 분석 결과 ===")
    for condition, score in result["condition_analysis"].items():
        print(f"{condition}: {score:.2f}")

    print("\n=== 성장 상태 메시지 ===")
    analyzer_message = analyzer.get_growth_status_message(result)
    print(analyzer_message)


if __name__ == "__main__":
    test_growth_analyzer()
