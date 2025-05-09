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
    crop_name_KOR TEXT,
    total_growth_days INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    max_days INTEGER
);

-- 성장 단계 테이블 (BBCH 스케일 반영)
CREATE TABLE growth_stages (
    stage_id INTEGER PRIMARY KEY,
    stage_name TEXT NOT NULL,
    stage_code TEXT NOT NULL,  -- BBCH 코드
    stage_description TEXT,
    days INTEGER NOT NULL,     -- 단계별 일수
    stage_features TEXT,       -- JSON 형식으로 단계별 특징 저장
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 기본 환경 조건 테이블
CREATE TABLE basic_conditions (
    condition_id INTEGER PRIMARY KEY,
    crop_id INTEGER NOT NULL,
    stage_id INTEGER NOT NULL,
    temperature_min REAL NOT NULL,
    temperature_max REAL NOT NULL,
    temperature_avg REAL NOT NULL,
    humidity_min REAL NOT NULL,
    humidity_max REAL NOT NULL,
    humidity_avg REAL NOT NULL,
    ph_min REAL NOT NULL,
    ph_max REAL NOT NULL,
    ph_avg REAL NOT NULL,
    rainfall_min REAL NOT NULL,
    rainfall_max REAL NOT NULL,
    rainfall_avg REAL NOT NULL,
    soil_moisture_min REAL NOT NULL,
    soil_moisture_max REAL NOT NULL,
    soil_moisture_avg REAL NOT NULL,
    sunlight_exposure_min REAL NOT NULL,
    sunlight_exposure_max REAL NOT NULL,
    sunlight_exposure_avg REAL NOT NULL,
    water_usage_efficiency_min REAL NOT NULL,
    water_usage_efficiency_max REAL NOT NULL,
    water_usage_efficiency_avg REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (crop_id) REFERENCES crops(crop_id),
    FOREIGN KEY (stage_id) REFERENCES growth_stages(stage_id)
);

-- 영양소 조건 테이블
CREATE TABLE nutrient_conditions (
    nutrient_id INTEGER PRIMARY KEY,
    crop_id INTEGER NOT NULL,
    stage_id INTEGER NOT NULL,
    nitrogen_min REAL NOT NULL,
    nitrogen_max REAL NOT NULL,
    nitrogen_avg REAL NOT NULL,
    phosphorus_min REAL NOT NULL,
    phosphorus_max REAL NOT NULL,
    phosphorus_avg REAL NOT NULL,
    potassium_min REAL NOT NULL,
    potassium_max REAL NOT NULL,
    potassium_avg REAL NOT NULL,
    chlorophyll_min REAL NOT NULL,
    chlorophyll_max REAL NOT NULL,
    chlorophyll_avg REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (crop_id) REFERENCES crops(crop_id),
    FOREIGN KEY (stage_id) REFERENCES growth_stages(stage_id)
);

-- 토양 조건 테이블
CREATE TABLE soil_conditions (
    soil_id INTEGER PRIMARY KEY,
    crop_id INTEGER NOT NULL,
    stage_id INTEGER NOT NULL,
    soil_type TEXT NOT NULL,
    organic_matter_content_min REAL NOT NULL,
    organic_matter_content_max REAL NOT NULL,
    organic_matter_content_avg REAL NOT NULL,
    irrigation_frequency_min INTEGER NOT NULL,
    irrigation_frequency_max INTEGER NOT NULL,
    irrigation_frequency_avg INTEGER NOT NULL,
    water_source_type TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (crop_id) REFERENCES crops(crop_id),
    FOREIGN KEY (stage_id) REFERENCES growth_stages(stage_id)
);

-- 스트레스 조건 테이블
CREATE TABLE stress_conditions (
    stress_id INTEGER PRIMARY KEY,
    crop_id INTEGER NOT NULL,
    stage_id INTEGER NOT NULL,
    wind_speed_min REAL NOT NULL,
    wind_speed_max REAL NOT NULL,
    wind_speed_avg REAL NOT NULL,
    co2_concentration_min REAL NOT NULL,
    co2_concentration_max REAL NOT NULL,
    co2_concentration_avg REAL NOT NULL,
    crop_density_min REAL NOT NULL,
    crop_density_max REAL NOT NULL,
    crop_density_avg REAL NOT NULL,
    pest_pressure_min REAL NOT NULL,
    pest_pressure_max REAL NOT NULL,
    pest_pressure_avg REAL NOT NULL,
    urban_area_proximity_min REAL NOT NULL,
    urban_area_proximity_max REAL NOT NULL,
    urban_area_proximity_avg REAL NOT NULL,
    frost_risk_min REAL NOT NULL,
    frost_risk_max REAL NOT NULL,
    frost_risk_avg REAL NOT NULL,
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
    soil_temperature REAL NOT NULL,
    ec REAL NOT NULL,
    chlorophyll_fluorescence REAL,
    ndvi REAL,
    evi REAL,
    root_activity REAL,
    is_outlier BOOLEAN DEFAULT FALSE,
    outlier_reason TEXT,
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
    predicted_progress REAL,
    prediction_error REAL,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (crop_id) REFERENCES crops(crop_id),
    FOREIGN KEY (stage_id) REFERENCES growth_stages(stage_id)
);

-- 성장 단계 데이터 삽입 (BBCH 스케일)
INSERT INTO growth_stages (stage_name, stage_code, stage_description, days, stage_features) VALUES
('germination', 'BBCH 00-09', '발아 단계: 씨앗 발아부터 지상부 출현까지', 7, 
 '{"soil_moisture_pattern": "급격한 증가", "temperature_sensitivity": "높음"}'),
('seedling', 'BBCH 10-19', '유묘 단계: 첫 잎 전개', 14,
 '{"leaf_development": "초기", "root_development": "활발"}'),
('early_vegetative', 'BBCH 20-29', '초기생장 단계: 주경 및 분얼 시작', 14,
 '{"tillering": "시작", "nitrogen_uptake": "증가"}'),
('tillering', 'BBCH 30-39', '분얼 단계: 줄기 신장 및 마디 형성', 14,
 '{"stem_elongation": "활발", "biomass_accumulation": "급증"}'),
('booting', 'BBCH 40-49', '이삭패밀 단계: 이삭 형성 시작', 14,
 '{"panicle_development": "시작", "nutrient_demand": "최대"}'),
('flowering', 'BBCH 50-69', '개화 단계: 이삭 추출과 개화', 14,
 '{"flowering_pattern": "순차적", "temperature_sensitivity": "매우 높음"}'),
('ripening', 'BBCH 70-89', '등숙 단계: 낟알 발달 및 성숙', 14,
 '{"grain_filling": "활발", "moisture_requirement": "감소"}'),
('maturity', 'BBCH 90-99', '완숙 단계: 수확 적기', 14,
 '{"moisture_content": "감소", "chlorophyll": "감소"}');

-- 벼 작물 데이터 삽입
INSERT INTO crops (crop_name, crop_name_KOR, total_growth_days) VALUES ('rice', '벼', 100);

