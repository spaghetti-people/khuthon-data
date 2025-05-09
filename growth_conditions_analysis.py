import pandas as pd
import numpy as np


def analyze_growth_conditions(csv_file):
    """
    작물의 성장 단계별 필요 조건을 분석하는 함수
    """
    # CSV 파일 읽기
    df = pd.read_csv(csv_file)

    # 성장 단계 레이블 추가
    df["stage_label"] = pd.cut(
        df["growth_stage"],
        bins=[0, 1, 2, 3],
        labels=["initial", "mid", "late"],
        include_lowest=True,
    )

    # 작물별 성장 기간 계산
    growth_periods = {}
    for crop in df["label"].unique():
        crop_data = df[df["label"] == crop]

        # 각 작물의 실제 성장 데이터에서 성장 기간 계산
        total_days = len(crop_data)  # 실제 데이터 포인트 수를 기반으로 계산

        # 각 단계별 데이터 수 계산
        stage_counts = crop_data["stage_label"].value_counts()

        # 각 단계별 실제 일수 계산
        initial_days = stage_counts.get("initial", 0)
        mid_days = stage_counts.get("mid", 0)
        late_days = stage_counts.get("late", 0)

        growth_periods[crop] = {
            "total_days": total_days,
            "stages": {"initial": initial_days, "mid": mid_days, "late": late_days},
        }

    # 분석할 환경 조건들
    conditions = [
        "temperature",
        "humidity",
        "ph",
        "rainfall",
        "soil_moisture",
        "sunlight_exposure",
        "water_usage_efficiency",
        "N",
        "P",
        "K",
        "soil_type",
        "wind_speed",
        "co2_concentration",
        "organic_matter",
        "irrigation_frequency",
        "crop_density",
        "pest_pressure",
        "urban_area_proximity",
        "water_source_type",
        "frost_risk",
    ]

    results = []

    # 각 작물별 분석
    for crop in df["label"].unique():
        crop_data = df[df["label"] == crop]

        # 각 성장 단계별 분석
        for stage in ["initial", "mid", "late"]:
            stage_data = crop_data[crop_data["stage_label"] == stage]

            if len(stage_data) == 0:
                continue

            # 각 조건별 적정 범위 계산
            stage_conditions = {
                "crop": crop,
                "growth_stage": stage,
                "data_count": len(stage_data),
            }

            for condition in conditions:
                values = stage_data[condition]

                # 25~75 퍼센타일을 적정 범위로 설정
                q1 = values.quantile(0.25)
                q3 = values.quantile(0.75)

                stage_conditions.update(
                    {
                        f"{condition}_min": round(q1, 2),
                        f"{condition}_max": round(q3, 2),
                        f"{condition}_avg": round(values.mean(), 2),
                    }
                )

            results.append(stage_conditions)

    # 결과를 데이터프레임으로 변환
    result_df = pd.DataFrame(results)

    # 게임에 활용할 수 있는 형태로 정리하여 출력
    print("\n=== 작물별 성장 단계 조건 분석 ===\n")

    stage_labels = {"initial": "초기", "mid": "중기", "late": "후기"}

    for crop in df["label"].unique():
        crop_conditions = result_df[result_df["crop"] == crop]
        print(f"\n[{crop} 성장 조건]")

        for stage in ["initial", "mid", "late"]:
            stage_cond = crop_conditions[crop_conditions["growth_stage"] == stage]
            if len(stage_cond) == 0:
                continue

            print(f"\n{stage_labels[stage]} 단계 조건:")
            print(f"1. 기본 환경 조건:")
            print(
                f"- 온도: {stage_cond['temperature_min'].iloc[0]}°C ~ {stage_cond['temperature_max'].iloc[0]}°C"
            )
            print(
                f"- 습도: {stage_cond['humidity_min'].iloc[0]}% ~ {stage_cond['humidity_max'].iloc[0]}%"
            )
            print(
                f"- pH: {stage_cond['ph_min'].iloc[0]} ~ {stage_cond['ph_max'].iloc[0]}"
            )
            print(
                f"- 강수량: {stage_cond['rainfall_min'].iloc[0]}mm ~ {stage_cond['rainfall_max'].iloc[0]}mm"
            )
            print(
                f"- 토양 수분: {stage_cond['soil_moisture_min'].iloc[0]}% ~ {stage_cond['soil_moisture_max'].iloc[0]}%"
            )
            print(
                f"- 일조량: {stage_cond['sunlight_exposure_min'].iloc[0]} ~ {stage_cond['sunlight_exposure_max'].iloc[0]}"
            )
            print(
                f"- 물 사용 효율: {stage_cond['water_usage_efficiency_min'].iloc[0]} ~ {stage_cond['water_usage_efficiency_max'].iloc[0]}"
            )

            print(f"\n2. 영양분 조건:")
            print(
                f"- 질소(N): {stage_cond['N_min'].iloc[0]} ~ {stage_cond['N_max'].iloc[0]}"
            )
            print(
                f"- 인(P): {stage_cond['P_min'].iloc[0]} ~ {stage_cond['P_max'].iloc[0]}"
            )
            print(
                f"- 칼륨(K): {stage_cond['K_min'].iloc[0]} ~ {stage_cond['K_max'].iloc[0]}"
            )

            print(f"\n3. 토양 및 수질 조건:")
            print(f"- 토양 유형: {stage_cond['soil_type_avg'].iloc[0]}")
            print(
                f"- 유기물 함량: {stage_cond['organic_matter_min'].iloc[0]} ~ {stage_cond['organic_matter_max'].iloc[0]}"
            )
            print(
                f"- 관개 빈도: {stage_cond['irrigation_frequency_min'].iloc[0]} ~ {stage_cond['irrigation_frequency_max'].iloc[0]}"
            )
            print(f"- 수원 유형: {stage_cond['water_source_type_avg'].iloc[0]}")

            print(f"\n4. 환경 스트레스 조건:")
            print(
                f"- 풍속: {stage_cond['wind_speed_min'].iloc[0]} ~ {stage_cond['wind_speed_max'].iloc[0]}"
            )
            print(
                f"- CO2 농도: {stage_cond['co2_concentration_min'].iloc[0]} ~ {stage_cond['co2_concentration_max'].iloc[0]}"
            )
            print(
                f"- 작물 밀도: {stage_cond['crop_density_min'].iloc[0]} ~ {stage_cond['crop_density_max'].iloc[0]}"
            )
            print(
                f"- 해충 압력: {stage_cond['pest_pressure_min'].iloc[0]} ~ {stage_cond['pest_pressure_max'].iloc[0]}"
            )
            print(
                f"- 도시 지역 근접도: {stage_cond['urban_area_proximity_min'].iloc[0]} ~ {stage_cond['urban_area_proximity_max'].iloc[0]}"
            )
            print(
                f"- 서리 위험: {stage_cond['frost_risk_min'].iloc[0]} ~ {stage_cond['frost_risk_max'].iloc[0]}"
            )

    # 결과를 여러 CSV 파일로 저장
    # 1. 전체 데이터 저장
    result_df.to_csv(
        "crop_growth_conditions_full.csv", index=False, encoding="utf-8-sig"
    )

    # 2. 기본 환경 조건만 저장
    basic_conditions = ["crop", "growth_stage", "data_count"] + [
        col
        for col in result_df.columns
        if any(
            x in col
            for x in [
                "temperature",
                "humidity",
                "ph",
                "rainfall",
                "soil_moisture",
                "sunlight_exposure",
                "water_usage_efficiency",
            ]
        )
    ]
    result_df[basic_conditions].to_csv(
        "crop_growth_conditions_basic.csv", index=False, encoding="utf-8-sig"
    )

    # 3. 영양분 조건만 저장
    nutrient_conditions = ["crop", "growth_stage", "data_count"] + [
        col for col in result_df.columns if any(x in col for x in ["N_", "P_", "K_"])
    ]
    result_df[nutrient_conditions].to_csv(
        "crop_growth_conditions_nutrients.csv", index=False, encoding="utf-8-sig"
    )

    # 4. 환경 스트레스 조건만 저장
    stress_conditions = ["crop", "growth_stage", "data_count"] + [
        col
        for col in result_df.columns
        if any(
            x in col
            for x in [
                "wind_speed",
                "co2_concentration",
                "crop_density",
                "pest_pressure",
                "urban_area_proximity",
                "frost_risk",
            ]
        )
    ]
    result_df[stress_conditions].to_csv(
        "crop_growth_conditions_stress.csv", index=False, encoding="utf-8-sig"
    )

    # 5. 토양 및 수질 조건만 저장
    soil_conditions = ["crop", "growth_stage", "data_count"] + [
        col
        for col in result_df.columns
        if any(
            x in col
            for x in [
                "soil_type",
                "organic_matter",
                "irrigation_frequency",
                "water_source_type",
            ]
        )
    ]
    result_df[soil_conditions].to_csv(
        "crop_growth_conditions_soil.csv", index=False, encoding="utf-8-sig"
    )

    print("\n분석 결과가 다음 파일들로 저장되었습니다:")
    print("1. crop_growth_conditions_full.csv - 전체 데이터")
    print("2. crop_growth_conditions_basic.csv - 기본 환경 조건")
    print("3. crop_growth_conditions_nutrients.csv - 영양분 조건")
    print("4. crop_growth_conditions_stress.csv - 환경 스트레스 조건")
    print("5. crop_growth_conditions_soil.csv - 토양 및 수질 조건")

    return growth_periods


if __name__ == "__main__":
    # 성장 조건 분석 실행
    conditions_df = analyze_growth_conditions("Crop_recommendationV2.csv")
