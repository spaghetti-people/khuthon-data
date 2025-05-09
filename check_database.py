import sqlite3
import pandas as pd


def check_database():
    # 데이터베이스 연결
    conn = sqlite3.connect("crop_conditions.db")

    try:
        # 테이블 목록 확인
        tables = pd.read_sql_query(
            """
            SELECT name FROM sqlite_master 
            WHERE type='table'
        """,
            conn,
        )
        print("\n=== 테이블 목록 ===")
        print(tables)

        # crops 테이블 확인
        crops = pd.read_sql_query("SELECT * FROM crops", conn)
        print("\n=== 작물 정보 ===")
        print(crops)

        # growth_stages 테이블 확인
        stages = pd.read_sql_query("SELECT * FROM growth_stages", conn)
        print("\n=== 성장 단계 정보 ===")
        print(stages)

        # basic_conditions 테이블 확인 (rice 작물)
        conditions = pd.read_sql_query(
            """
            SELECT bc.*, c.crop_name, gs.stage_name
            FROM basic_conditions bc
            JOIN crops c ON bc.crop_id = c.crop_id
            JOIN growth_stages gs ON bc.stage_id = gs.stage_id
            WHERE c.crop_name = 'rice'
        """,
            conn,
        )
        print("\n=== 벼 작물의 기본 환경 조건 ===")
        print(conditions)

    except Exception as e:
        print(f"오류 발생: {str(e)}")

    finally:
        conn.close()


if __name__ == "__main__":
    check_database()
