OLD_DATASET_NAMES = [
    "airpassengers",
    "australian_electricity_demand",
    "car_parts_without_missing",
    "cif_2016",
    "covid_deaths",
    "electricity",
    "electricity_weekly",
    "exchange_rate",
    "fred_md",
    "hospital",
    "kaggle_web_traffic_weekly",
    "kdd_cup_2018_without_missing",
    "london_smart_meters_without_missing",
    "nn5_daily_with_missing",
    "nn5_weekly",
    "pedestrian_counts",
    "rideshare_without_missing",
    "saugeenday",
    "solar-energy",
    "solar_10_minutes",
    "solar_weekly",
    "taxi_30min",
    "temperature_rain_without_missing",
    "tourism_monthly",
    "uber_tlc_daily",
    "uber_tlc_hourly",
    "vehicle_trips_without_missing",
    "weather",
    "wiki-rolling_nips",
    "m4_daily",
    "m4_weekly",
    "m4_hourly",
    "m4_monthly",
    "m4_quarterly",
    "m4_yearly",
    "traffic",
    "wind_farms_without_missing",
]

ENERGY_DOMAIN = [
    "australian_electricity_demand",  # Used
    "electricity",  # Used # Has rolling-window test
    "electricity_weekly",  # AutoGluon baseline available
    "electricity_hourly",  # Used # AutoGluon baseline available
    "london_smart_meters_without_missing",  # Used
    "solar-energy",  # Used # Has rolling-window test
    "solar_10_minutes",  # Used
    "solar_weekly",
    "wind_farms_without_missing",  # Used
    "solar_nips",  # Used # Has rolling-window test
]

TRANSPORT_DOMAIN = [
    "airpassengers",
    "pedestrian_counts",  # Used # AutoGluon baseline available
    "rideshare_without_missing",
    "taxi_30min",
    "uber_tlc_daily",
    "uber_tlc_hourly",  # Used
    "vehicle_trips_without_missing",  # AutoGluon baseline available
    "traffic",  # Used # Has rolling-window test
    "traffic_nips",  # Used # Has rolling-window test
]

MULTIPLE_DOMAIN = [
    "m1_monthly",  # AutoGluon baseline available
    "m1_quarterly",  # AutoGluon baseline available
    "m1_yearly",  # AutoGluon baseline available
    "m3_monthly",  # AutoGluon baseline available
    "m3_quarterly",  # AutoGluon baseline available
    "m3_yearly",  # AutoGluon baseline available
    "m3_other",  # AutoGluon baseline available
    "m4_daily",  # AutoGluon baseline available
    "m4_weekly",  # Used # AutoGluon baseline available
    "m4_hourly",  # AutoGluon baseline available
    "m4_monthly",  # AutoGluon baseline available
    "m4_quarterly",  # AutoGluon baseline available
    "m4_yearly",  # AutoGluon baseline available
]

NATURE_DOMAIN = [
    "covid_deaths",  # AutoGluon baseline available
    "kdd_cup_2018_without_missing",  # Used # AutoGluon baseline available
    "saugeenday",  # Used
    "temperature_rain_without_missing",
    "weather",  # Used
    "sunspot_without_missing",  # Used
]
ECONOMIC_DOMAIN = [
    "exchange_rate",  # Used # Has rolling-window test
    "fred_md",  # AutoGluon baseline available
]

BANKING_DOMAIN = [
    "cif_2016",  # AutoGluon baseline available
    "nn5_daily_without_missing",  # 700-ish, could be used after adaptation # AutoGluon baseline available
    "nn5_weekly",  # AutoGluon baseline available
]
SALES_DOMAIN = [
    "car_parts_without_missing",  # AutoGluon baseline available
    "dominick",
]

WEB_DOMAIN = [
    "kaggle_web_traffic_without_missing",  # 700-ish, could be used after adaptation
    "kaggle_web_traffic_weekly",  # AutoGluon baseline available
]

HEALTH_DOMAIN = [
    "hospital",  # AutoGluon baseline available
]

TOURISM_DOMAIN = ["tourism_monthly", "tourism_yearly", "tourism_quarterly"]

DATASET_NAMES = (
    ENERGY_DOMAIN
    + TRANSPORT_DOMAIN
    + MULTIPLE_DOMAIN
    + NATURE_DOMAIN
    + ECONOMIC_DOMAIN
    + BANKING_DOMAIN
    + SALES_DOMAIN
    + WEB_DOMAIN
    + HEALTH_DOMAIN
)

# Datasets (1): timesteps in train and test >= 1200; no anomalies

FILTERED_ENERGY_DOMAIN_1 = [
    "australian_electricity_demand",
    "electricity_hourly",
    "london_smart_meters_without_missing",
    "solar_10_minutes",
    "wind_farms_without_missing",
]
FILTERED_TRANSPORT_DOMAIN_1 = ["pedestrian_counts", "uber_tlc_hourly", "traffic"]
FILTERED_NATURE_DOMAIN_1 = [
    "kdd_cup_2018_without_missing",
    "saugeenday",
    "sunspot_without_missing",
]
FILTERED_ECONOMIC_DOMAIN_1 = ["exchange_rate"]
NEW_DATASETS_1 = ["ett_h1", "ett_h2", "ett_m1", "ett_m2", "beijing_pm25", "AirQualityUCI", "beijing_multisite", "cpu_limit_minute", "cpu_usage_minute", \
                "function_delay_minute", "instances_minute", \
                "memory_limit_minute", "memory_usage_minute", \
                "platform_delay_minute", "requests_minute"]
DATASET_NAMES_1 = (
    FILTERED_ENERGY_DOMAIN_1
    + FILTERED_TRANSPORT_DOMAIN_1
    + FILTERED_NATURE_DOMAIN_1
    + FILTERED_ECONOMIC_DOMAIN_1
    + NEW_DATASETS_1
)

