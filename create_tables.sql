-- 작물 테이블
CREATE TABLE crops (
    crop_id INTEGER PRIMARY KEY,
    crop_name TEXT NOT NULL UNIQUE,
    total_growth_days INTEGER NOT NULL,  -- 총 성장 기간 (일)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 성장 단계 테이블
CREATE TABLE growth_stages (
    stage_id INTEGER PRIMARY KEY,
    crop_id INTEGER NOT NULL,
    stage_name TEXT NOT NULL,
    days INTEGER NOT NULL,  -- 단계별 일수
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (crop_id) REFERENCES crops(crop_id)
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

-- 영양분 조건 테이블
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
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (crop_id) REFERENCES crops(crop_id),
    FOREIGN KEY (stage_id) REFERENCES growth_stages(stage_id)
);

-- 토양 조건 테이블
CREATE TABLE soil_conditions (
    soil_id INTEGER PRIMARY KEY,
    crop_id INTEGER NOT NULL,
    stage_id INTEGER NOT NULL,
    soil_type INTEGER NOT NULL,
    organic_matter_content_min REAL NOT NULL,
    organic_matter_content_max REAL NOT NULL,
    organic_matter_content_avg REAL NOT NULL,
    irrigation_frequency_min REAL NOT NULL,
    irrigation_frequency_max REAL NOT NULL,
    irrigation_frequency_avg REAL NOT NULL,
    water_source_type INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (crop_id) REFERENCES crops(crop_id),
    FOREIGN KEY (stage_id) REFERENCES growth_stages(stage_id)
);

-- 환경 스트레스 조건 테이블
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

-- 작물 성장 기록 테이블
CREATE TABLE growth_records (
    record_id INTEGER PRIMARY KEY,
    crop_id INTEGER NOT NULL,
    planting_date DATE NOT NULL,  -- 파종일
    current_date DATE NOT NULL,   -- 현재 날짜
    current_stage_id INTEGER NOT NULL,
    growth_progress REAL NOT NULL,  -- 성장 진행률 (%)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (crop_id) REFERENCES crops(crop_id),
    FOREIGN KEY (current_stage_id) REFERENCES growth_stages(stage_id)
); 