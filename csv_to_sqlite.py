import pandas as pd
import sqlite3
from datetime import datetime
from growth_conditions_analysis import analyze_growth_conditions


def create_database():
    # 성장 조건 분석 실행
    growth_periods = analyze_growth_conditions("Crop_recommendationV2.csv")

    # CSV 파일 읽기
    basic_df = pd.read_csv("crop_growth_conditions_basic.csv")
    nutrient_df = pd.read_csv("crop_growth_conditions_nutrients.csv")
    soil_df = pd.read_csv("crop_growth_conditions_soil.csv")
    stress_df = pd.read_csv("crop_growth_conditions_stress.csv")

    # 데이터베이스 연결
    conn = sqlite3.connect("crop_conditions.db")
    cursor = conn.cursor()

    try:
        # 기존 테이블 삭제
        cursor.executescript(
            """
            DROP TABLE IF EXISTS growth_records;
            DROP TABLE IF EXISTS stress_conditions;
            DROP TABLE IF EXISTS soil_conditions;
            DROP TABLE IF EXISTS nutrient_conditions;
            DROP TABLE IF EXISTS basic_conditions;
            DROP TABLE IF EXISTS growth_stages;
            DROP TABLE IF EXISTS crops;
        """
        )

        # create_tables.sql 실행
        with open("create_tables.sql", "r") as sql_file:
            cursor.executescript(sql_file.read())

        # 작물 데이터 삽입
        for crop, period in growth_periods.items():
            print(f"작물: {crop}, 성장 기간: {period['total_days']}일")
            cursor.execute(
                """
                INSERT INTO crops (crop_name, total_growth_days, created_at)
                VALUES (?, ?, datetime('now'))
                """,
                (crop, period["total_days"]),
            )
            crop_id = cursor.lastrowid

            # 성장 단계 데이터 삽입
            for stage_name, days in period["stages"].items():
                cursor.execute(
                    """
                    INSERT INTO growth_stages (crop_id, stage_name, days, created_at)
                    VALUES (?, ?, ?, datetime('now'))
                    """,
                    (crop_id, stage_name, days),
                )

        # 기본 환경 조건 데이터 삽입
        for _, row in basic_df.iterrows():
            cursor.execute(
                """
                SELECT crop_id FROM crops WHERE crop_name = ?
            """,
                (row["crop"],),
            )
            crop_id = cursor.fetchone()[0]

            cursor.execute(
                """
                SELECT stage_id FROM growth_stages WHERE stage_name = ?
            """,
                (row["growth_stage"],),
            )
            stage_id = cursor.fetchone()[0]

            cursor.execute(
                """
                INSERT INTO basic_conditions (
                    crop_id, stage_id, temperature_min, temperature_max, temperature_avg,
                    humidity_min, humidity_max, humidity_avg, ph_min, ph_max, ph_avg,
                    rainfall_min, rainfall_max, rainfall_avg, soil_moisture_min,
                    soil_moisture_max, soil_moisture_avg, sunlight_exposure_min,
                    sunlight_exposure_max, sunlight_exposure_avg, water_usage_efficiency_min,
                    water_usage_efficiency_max, water_usage_efficiency_avg, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """,
                (
                    crop_id,
                    stage_id,
                    row["temperature_min"],
                    row["temperature_max"],
                    row["temperature_avg"],
                    row["humidity_min"],
                    row["humidity_max"],
                    row["humidity_avg"],
                    row["ph_min"],
                    row["ph_max"],
                    row["ph_avg"],
                    row["rainfall_min"],
                    row["rainfall_max"],
                    row["rainfall_avg"],
                    row["soil_moisture_min"],
                    row["soil_moisture_max"],
                    row["soil_moisture_avg"],
                    row["sunlight_exposure_min"],
                    row["sunlight_exposure_max"],
                    row["sunlight_exposure_avg"],
                    row["water_usage_efficiency_min"],
                    row["water_usage_efficiency_max"],
                    row["water_usage_efficiency_avg"],
                ),
            )

        # 영양분 조건 데이터 삽입
        for _, row in nutrient_df.iterrows():
            cursor.execute(
                """
                SELECT crop_id FROM crops WHERE crop_name = ?
            """,
                (row["crop"],),
            )
            crop_id = cursor.fetchone()[0]

            cursor.execute(
                """
                SELECT stage_id FROM growth_stages WHERE stage_name = ?
            """,
                (row["growth_stage"],),
            )
            stage_id = cursor.fetchone()[0]

            cursor.execute(
                """
                INSERT INTO nutrient_conditions (
                    crop_id, stage_id, nitrogen_min, nitrogen_max, nitrogen_avg,
                    phosphorus_min, phosphorus_max, phosphorus_avg,
                    potassium_min, potassium_max, potassium_avg, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """,
                (
                    crop_id,
                    stage_id,
                    row["N_min"],
                    row["N_max"],
                    row["N_avg"],
                    row["P_min"],
                    row["P_max"],
                    row["P_avg"],
                    row["K_min"],
                    row["K_max"],
                    row["K_avg"],
                ),
            )

        # 토양 조건 데이터 삽입
        for _, row in soil_df.iterrows():
            cursor.execute(
                """
                SELECT crop_id FROM crops WHERE crop_name = ?
            """,
                (row["crop"],),
            )
            crop_id = cursor.fetchone()[0]

            cursor.execute(
                """
                SELECT stage_id FROM growth_stages WHERE stage_name = ?
            """,
                (row["growth_stage"],),
            )
            stage_id = cursor.fetchone()[0]

            cursor.execute(
                """
                INSERT INTO soil_conditions (
                    crop_id, stage_id, soil_type, organic_matter_content_min,
                    organic_matter_content_max, organic_matter_content_avg,
                    irrigation_frequency_min, irrigation_frequency_max,
                    irrigation_frequency_avg, water_source_type, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """,
                (
                    crop_id,
                    stage_id,
                    row["soil_type_avg"],
                    row["organic_matter_min"],
                    row["organic_matter_max"],
                    row["organic_matter_avg"],
                    row["irrigation_frequency_min"],
                    row["irrigation_frequency_max"],
                    row["irrigation_frequency_avg"],
                    row["water_source_type_avg"],
                ),
            )

        # 환경 스트레스 조건 데이터 삽입
        for _, row in stress_df.iterrows():
            cursor.execute(
                """
                SELECT crop_id FROM crops WHERE crop_name = ?
            """,
                (row["crop"],),
            )
            crop_id = cursor.fetchone()[0]

            cursor.execute(
                """
                SELECT stage_id FROM growth_stages WHERE stage_name = ?
            """,
                (row["growth_stage"],),
            )
            stage_id = cursor.fetchone()[0]

            cursor.execute(
                """
                INSERT INTO stress_conditions (
                    crop_id, stage_id, wind_speed_min, wind_speed_max, wind_speed_avg,
                    co2_concentration_min, co2_concentration_max, co2_concentration_avg,
                    crop_density_min, crop_density_max, crop_density_avg,
                    pest_pressure_min, pest_pressure_max, pest_pressure_avg,
                    urban_area_proximity_min, urban_area_proximity_max, urban_area_proximity_avg,
                    frost_risk_min, frost_risk_max, frost_risk_avg, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """,
                (
                    crop_id,
                    stage_id,
                    row["wind_speed_min"],
                    row["wind_speed_max"],
                    row["wind_speed_avg"],
                    row["co2_concentration_min"],
                    row["co2_concentration_max"],
                    row["co2_concentration_avg"],
                    row["crop_density_min"],
                    row["crop_density_max"],
                    row["crop_density_avg"],
                    row["pest_pressure_min"],
                    row["pest_pressure_max"],
                    row["pest_pressure_avg"],
                    row["urban_area_proximity_min"],
                    row["urban_area_proximity_max"],
                    row["urban_area_proximity_avg"],
                    row["frost_risk_min"],
                    row["frost_risk_max"],
                    row["frost_risk_avg"],
                ),
            )

        conn.commit()
        print("데이터베이스가 성공적으로 생성되었습니다.")

    except Exception as e:
        print(f"오류가 발생했습니다: {str(e)}")
        conn.rollback()
    finally:
        conn.close()


def insert_crop_data(conn):
    # 성장 조건 분석 실행
    growth_periods = analyze_growth_conditions("Crop_recommendationV2.csv")

    # CSV 파일들 읽기
    basic_df = pd.read_csv("crop_growth_conditions_basic.csv")
    nutrient_df = pd.read_csv("crop_growth_conditions_nutrients.csv")
    soil_df = pd.read_csv("crop_growth_conditions_soil.csv")
    stress_df = pd.read_csv("crop_growth_conditions_stress.csv")

    cursor = conn.cursor()

    # 작물 데이터 삽입
    for crop, period in growth_periods.items():
        # 이미 존재하는 작물인지 확인
        cursor.execute("SELECT crop_id FROM crops WHERE crop_name = ?", (crop,))
        crop_id = cursor.fetchone()
        if crop_id is None:  # 존재하지 않는 경우에만 삽입
            cursor.execute(
                "INSERT INTO crops (crop_name, total_growth_days) VALUES (?, ?)",
                (crop, period["total_days"]),
            )
            crop_id = cursor.lastrowid
        else:
            crop_id = crop_id[0]

        # 성장 단계 데이터 삽입
        for stage_name, days in period["stages"].items():
            cursor.execute(
                "INSERT INTO growth_stages (crop_id, stage_name, days) VALUES (?, ?, ?)",
                (crop_id, stage_name, days),
            )

    conn.commit()

    # 각 테이블에 데이터 삽입
    for _, row in basic_df.iterrows():
        # 작물 ID 가져오기
        cursor.execute("SELECT crop_id FROM crops WHERE crop_name = ?", (row["crop"],))
        crop_id = cursor.fetchone()[0]

        # 성장 단계 ID 가져오기
        cursor.execute(
            "SELECT stage_id FROM growth_stages WHERE crop_id = ? AND stage_name = ?",
            (crop_id, row["growth_stage"]),
        )
        stage_id = cursor.fetchone()[0]

        # 기본 환경 조건 삽입
        cursor.execute(
            """
            INSERT INTO basic_conditions (
                crop_id, stage_id, temperature_min, temperature_max, temperature_avg,
                humidity_min, humidity_max, humidity_avg, ph_min, ph_max, ph_avg,
                rainfall_min, rainfall_max, rainfall_avg, soil_moisture_min,
                soil_moisture_max, soil_moisture_avg, sunlight_exposure_min,
                sunlight_exposure_max, sunlight_exposure_avg, water_usage_efficiency_min,
                water_usage_efficiency_max, water_usage_efficiency_avg
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                crop_id,
                stage_id,
                row["temperature_min"],
                row["temperature_max"],
                row["temperature_avg"],
                row["humidity_min"],
                row["humidity_max"],
                row["humidity_avg"],
                row["ph_min"],
                row["ph_max"],
                row["ph_avg"],
                row["rainfall_min"],
                row["rainfall_max"],
                row["rainfall_avg"],
                row["soil_moisture_min"],
                row["soil_moisture_max"],
                row["soil_moisture_avg"],
                row["sunlight_exposure_min"],
                row["sunlight_exposure_max"],
                row["sunlight_exposure_avg"],
                row["water_usage_efficiency_min"],
                row["water_usage_efficiency_max"],
                row["water_usage_efficiency_avg"],
            ),
        )

    # 영양분 조건 삽입
    for _, row in nutrient_df.iterrows():
        cursor.execute("SELECT crop_id FROM crops WHERE crop_name = ?", (row["crop"],))
        crop_id = cursor.fetchone()[0]
        cursor.execute(
            "SELECT stage_id FROM growth_stages WHERE crop_id = ? AND stage_name = ?",
            (crop_id, row["growth_stage"]),
        )
        stage_id = cursor.fetchone()[0]

        cursor.execute(
            """
            INSERT INTO nutrient_conditions (
                crop_id, stage_id, nitrogen_min, nitrogen_max, nitrogen_avg,
                phosphorus_min, phosphorus_max, phosphorus_avg,
                potassium_min, potassium_max, potassium_avg
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                crop_id,
                stage_id,
                row["N_min"],
                row["N_max"],
                row["N_avg"],
                row["P_min"],
                row["P_max"],
                row["P_avg"],
                row["K_min"],
                row["K_max"],
                row["K_avg"],
            ),
        )

    # 토양 조건 삽입
    for _, row in soil_df.iterrows():
        cursor.execute("SELECT crop_id FROM crops WHERE crop_name = ?", (row["crop"],))
        crop_id = cursor.fetchone()[0]
        cursor.execute(
            "SELECT stage_id FROM growth_stages WHERE crop_id = ? AND stage_name = ?",
            (crop_id, row["growth_stage"]),
        )
        stage_id = cursor.fetchone()[0]

        cursor.execute(
            """
            INSERT INTO soil_conditions (
                crop_id, stage_id, soil_type, organic_matter_content_min,
                organic_matter_content_max, organic_matter_content_avg,
                irrigation_frequency_min, irrigation_frequency_max,
                irrigation_frequency_avg, water_source_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                crop_id,
                stage_id,
                row["soil_type_avg"],
                row["organic_matter_min"],
                row["organic_matter_max"],
                row["organic_matter_avg"],
                row["irrigation_frequency_min"],
                row["irrigation_frequency_max"],
                row["irrigation_frequency_avg"],
                row["water_source_type_avg"],
            ),
        )

    # 환경 스트레스 조건 삽입
    for _, row in stress_df.iterrows():
        cursor.execute("SELECT crop_id FROM crops WHERE crop_name = ?", (row["crop"],))
        crop_id = cursor.fetchone()[0]
        cursor.execute(
            "SELECT stage_id FROM growth_stages WHERE crop_id = ? AND stage_name = ?",
            (crop_id, row["growth_stage"]),
        )
        stage_id = cursor.fetchone()[0]

        cursor.execute(
            """
            INSERT INTO stress_conditions (
                crop_id, stage_id, wind_speed_min, wind_speed_max, wind_speed_avg,
                co2_concentration_min, co2_concentration_max, co2_concentration_avg,
                crop_density_min, crop_density_max, crop_density_avg,
                pest_pressure_min, pest_pressure_max, pest_pressure_avg,
                urban_area_proximity_min, urban_area_proximity_max, urban_area_proximity_avg,
                frost_risk_min, frost_risk_max, frost_risk_avg
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                crop_id,
                stage_id,
                row["wind_speed_min"],
                row["wind_speed_max"],
                row["wind_speed_avg"],
                row["co2_concentration_min"],
                row["co2_concentration_max"],
                row["co2_concentration_avg"],
                row["crop_density_min"],
                row["crop_density_max"],
                row["crop_density_avg"],
                row["pest_pressure_min"],
                row["pest_pressure_max"],
                row["pest_pressure_avg"],
                row["urban_area_proximity_min"],
                row["urban_area_proximity_max"],
                row["urban_area_proximity_avg"],
                row["frost_risk_min"],
                row["frost_risk_max"],
                row["frost_risk_avg"],
            ),
        )

    conn.commit()


def main():
    try:
        # 데이터베이스 연결
        conn = sqlite3.connect("crop_conditions.db")

        # 데이터베이스 생성
        create_database()
        print("데이터베이스 테이블이 생성되었습니다.")

        # 데이터 삽입
        insert_crop_data(conn)
        print("CSV 데이터가 성공적으로 데이터베이스에 삽입되었습니다.")

        # 연결 종료
        conn.close()
        print("데이터베이스 연결이 종료되었습니다.")

    except Exception as e:
        print(f"오류가 발생했습니다: {str(e)}")


if __name__ == "__main__":
    main()
