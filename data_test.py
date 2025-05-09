import json
from datetime import datetime, date, timedelta
from growth_analyzer import GrowthAnalyzer


def test_daily_growth():
    # JSON 파일 읽기
    with open("test_data.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # 분석기 초기화
    analyzer = GrowthAnalyzer()

    # 파종일 파싱
    planting_date = datetime.strptime(test_data["planting_date"], "%Y-%m-%d").date()

    print(f"\n=== {test_data['crop_name']} 작물 성장 분석 시작 ===")
    print(f"파종일: {planting_date}")

    # 각 날짜별 분석 실행
    for day_data in test_data["daily_conditions"]:
        day = day_data["day"]
        raw_conditions = day_data["conditions"]
        current_date = planting_date + timedelta(days=day)

        # 센서 데이터 키 이름 변환
        conditions = {
            "temperature": raw_conditions["temperature"],
            "humidity": raw_conditions["humidity"],
            "ph": raw_conditions["ph"],
            "rainfall": raw_conditions["rainfall"],
            "soil_moisture": raw_conditions["soil_moisture"],
            "sunlight_exposure": raw_conditions["sunlight_exposure"],
            "co2_concentration": int(raw_conditions["co2_concentration"]),
            "nitrogen": raw_conditions["N"],
            "phosphorus": raw_conditions["P"],
            "potassium": raw_conditions["K"],
            "irrigation_frequency": 2,  # 기본값 설정
            "fertilizer_usage": 1.5,  # 기본값 설정
        }

        print(f"\n--- {day}일차 분석 ---")
        print(f"현재 날짜: {current_date}")

        try:
            # 성장 분석 실행
            result = analyzer.analyze_growth(
                crop_name=test_data["crop_name"],
                current_conditions=conditions,
                planting_date=planting_date,
                current_date=current_date,
            )

            # 결과 출력
            print(f"현재 성장 단계: {result['current_stage']}")
            print(f"성장 진행도: {result['growth_progress']:.1f}%")
            print(f"남은 성장도: {result['remaining_progress']:.1f}%")
            print(f"파종 후 경과일: {result['days_since_planting']}일")

            print("\n조건별 분석 결과:")
            for condition, score in result["condition_analysis"].items():
                print(f"- {condition}: {score:.2f}")

            print("\n성장 상태 메시지:")
            print(analyzer.get_growth_status_message(result))

        except Exception as e:
            print(f"오류 발생: {str(e)}")

    # 분석기 종료
    analyzer.close()


if __name__ == "__main__":
    test_daily_growth()
