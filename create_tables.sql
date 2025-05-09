-- 기존 테이블 삭제
DROP TABLE IF EXISTS sensor_data;
DROP TABLE IF EXISTS management_conditions;
DROP TABLE IF EXISTS growth_records;
DROP TABLE IF EXISTS stress_conditions;
DROP TABLE IF EXISTS soil_conditions;
DROP TABLE IF EXISTS nutrient_conditions;
DROP TABLE IF EXISTS basic_conditions;
DROP TABLE IF EXISTS growth_stages;
DROP TABLE IF EXISTS crops;

-- 작물 정보 테이블
CREATE TABLE crops (
    crop_id INTEGER PRIMARY KEY,
    crop_name TEXT NOT NULL,
    total_growth_days INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 성장 단계 테이블
CREATE TABLE growth_stages (
    stage_id INTEGER PRIMARY KEY,
    stage_name TEXT NOT NULL,
    stage_description TEXT,
    min_days INTEGER NOT NULL,
    max_days INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 기본 환경 조건 테이블
CREATE TABLE basic_conditions (
    condition_id INTEGER PRIMARY KEY,
    crop_id INTEGER NOT NULL,
    stage_id INTEGER NOT NULL,
    min_temperature REAL NOT NULL,
    max_temperature REAL NOT NULL,
    min_humidity REAL NOT NULL,
    max_humidity REAL NOT NULL,
    min_ph REAL NOT NULL,
    max_ph REAL NOT NULL,
    min_rainfall REAL NOT NULL,
    max_rainfall REAL NOT NULL,
    min_co2_concentration INTEGER NOT NULL,
    max_co2_concentration INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (crop_id) REFERENCES crops(crop_id),
    FOREIGN KEY (stage_id) REFERENCES growth_stages(stage_id)
);

-- 영양소 조건 테이블
CREATE TABLE nutrient_conditions (
    nutrient_id INTEGER PRIMARY KEY,
    crop_id INTEGER NOT NULL,
    stage_id INTEGER NOT NULL,
    min_nitrogen REAL NOT NULL,
    max_nitrogen REAL NOT NULL,
    min_phosphorus REAL NOT NULL,
    max_phosphorus REAL NOT NULL,
    min_potassium REAL NOT NULL,
    max_potassium REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (crop_id) REFERENCES crops(crop_id),
    FOREIGN KEY (stage_id) REFERENCES growth_stages(stage_id)
);

-- 토양 조건 테이블
CREATE TABLE soil_conditions (
    soil_id INTEGER PRIMARY KEY,
    crop_id INTEGER NOT NULL,
    stage_id INTEGER NOT NULL,
    min_soil_moisture REAL NOT NULL,
    max_soil_moisture REAL NOT NULL,
    soil_type TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (crop_id) REFERENCES crops(crop_id),
    FOREIGN KEY (stage_id) REFERENCES growth_stages(stage_id)
);

-- 스트레스 조건 테이블
CREATE TABLE stress_conditions (
    stress_id INTEGER PRIMARY KEY,
    crop_id INTEGER NOT NULL,
    stage_id INTEGER NOT NULL,
    max_wind_speed REAL NOT NULL,
    max_pest_pressure INTEGER NOT NULL,
    max_urban_area_proximity INTEGER NOT NULL,
    max_frost_risk INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (crop_id) REFERENCES crops(crop_id),
    FOREIGN KEY (stage_id) REFERENCES growth_stages(stage_id)
);

-- 관리 조건 테이블
CREATE TABLE management_conditions (
    management_id INTEGER PRIMARY KEY,
    crop_id INTEGER NOT NULL,
    stage_id INTEGER NOT NULL,
    irrigation_frequency INTEGER NOT NULL,
    fertilizer_usage REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (crop_id) REFERENCES crops(crop_id),
    FOREIGN KEY (stage_id) REFERENCES growth_stages(stage_id)
);

-- 센서 데이터 테이블
CREATE TABLE sensor_data (
    data_id INTEGER PRIMARY KEY,
    crop_id INTEGER NOT NULL,
    temperature REAL NOT NULL,
    humidity REAL NOT NULL,
    ph REAL NOT NULL,
    rainfall REAL NOT NULL,
    soil_moisture REAL NOT NULL,
    sunlight_exposure REAL NOT NULL,
    co2_concentration INTEGER NOT NULL,
    nitrogen REAL NOT NULL,
    phosphorus REAL NOT NULL,
    potassium REAL NOT NULL,
    irrigation_frequency INTEGER NOT NULL,
    fertilizer_usage REAL NOT NULL,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (crop_id) REFERENCES crops(crop_id)
);

-- 성장 기록 테이블
CREATE TABLE growth_records (
    record_id INTEGER PRIMARY KEY,
    crop_id INTEGER NOT NULL,
    stage_id INTEGER NOT NULL,
    growth_progress REAL NOT NULL,
    condition_score REAL NOT NULL,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (crop_id) REFERENCES crops(crop_id),
    FOREIGN KEY (stage_id) REFERENCES growth_stages(stage_id)
);

-- 성장 단계 데이터 삽입
INSERT INTO growth_stages (stage_name, stage_description, min_days, max_days) VALUES
('germination', '발아 단계: 씨앗이 발아하여 새싹이 나오는 단계', 0, 7),
('seedling', '유묘 단계: 새싹이 자라나 잎이 나오는 단계', 8, 21),
('vegetative', '영양생장 단계: 잎과 줄기가 활발히 자라는 단계', 22, 49),
('flowering', '개화 단계: 꽃이 피고 열매가 맺히는 단계', 50, 77),
('maturity', '성숙 단계: 열매가 완전히 익는 단계', 78, 100);

-- 벼 작물 데이터 삽입
INSERT INTO crops (crop_name, total_growth_days) VALUES ('rice', 100); 