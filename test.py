from growth_analyzer import GrowthAnalyzer
from datetime import datetime, timedelta


def test_growth_analyzer():
    # 분석기 초기화
    analyzer = GrowthAnalyzer()

    # 테스트할 작물과 날짜 설정
    crop_name = "rice"
    planting_date = datetime.now() - timedelta(days=15)  # 15일 전에 심은 벼

    # 현재 센서 데이터 (예시)
    current_conditions = {
        "temperature": 24.5,
        "humidity": 82.0,
        "ph": 6.0,
        "rainfall": 120.0,
        "soil_moisture": 25.0,
        "sunlight_exposure": 9.0,
        "water_usage_efficiency": 3.0,
    }

    # 성장 상태 분석
    result = analyzer.analyze_growth_condition(
        crop_name=crop_name,
        current_conditions=current_conditions,
        planting_date=planting_date,
    )

    # 결과 출력
    print("\n=== 작물 성장 상태 분석 결과 ===")
    print(f"작물: {crop_name}")
    print(f"파종일: {planting_date.strftime('%Y-%m-%d')}")
    print(f"현재 성장 단계: {result['current_stage']}")
    print(f"성장 진행률: {result['growth_progress']:.1f}%")
    print(f"전체 점수: {result['total_score']:.1f}/100")

    print("\n=== 세부 점수 ===")
    for condition, score in result["detailed_scores"].items():
        print(f"{condition}: {score:.1f}/100")

    print("\n=== 개선 제안 ===")
    for suggestion in result["improvement_suggestions"]:
        print(f"- {suggestion}")


if __name__ == "__main__":
    test_growth_analyzer()
