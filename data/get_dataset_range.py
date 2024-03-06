"""
Not used in project. Just used to print the start and end dates of each datasets and some other statistics, for our reference.
"""
from pathlib import Path

from gluonts.dataset.repository.datasets import get_dataset
from pandas.tseries.frequencies import to_offset

ENERGY_DOMAIN = [
    "australian_electricity_demand",
    "electricity",
    "electricity_hourly",
    "london_smart_meters_without_missing",
    "solar-energy",
    "solar_10_minutes",
    "wind_farms_without_missing",
    "solar_nips",
]

TRANSPORT_DOMAIN = ["pedestrian_counts", "uber_tlc_hourly", "traffic", "traffic_nips"]

NATURE_DOMAIN = [
    "kdd_cup_2018_without_missing",
    "saugeenday",
    "temperature_rain_without_missing",
    "weather",
    "sunspot_without_missing",
]

ECONOMIC_DOMAIN = [
    "exchange_rate",
]

BANKING_DOMAIN = [
    "nn5_daily_without_missing",
]

WEB_DOMAIN = [
    "kaggle_web_traffic_without_missing",
]

DATASET_NAMES = (
    ENERGY_DOMAIN
    + TRANSPORT_DOMAIN
    + NATURE_DOMAIN
    + ECONOMIC_DOMAIN
    + BANKING_DOMAIN
    + WEB_DOMAIN
)

# Change to your dataset_path
dataset_path = Path("/home/toolkit/datasets")

for dataset_name in DATASET_NAMES:
    dataset = get_dataset(dataset_name, path=dataset_path)

    min_train_start_date = None
    max_train_end_date = None

    # Iterate through the dataset and get start and end dates
    for entry in dataset.train:
        start_date = entry["start"]
        length_of_series = len(entry["target"])
        frequency = dataset.metadata.freq

        # Calculate the end date
        end_date = start_date + to_offset(frequency) * (length_of_series - 1)

        if min_train_start_date is None or start_date < min_train_start_date:
            min_train_start_date = start_date

        if max_train_end_date is None or end_date > max_train_end_date:
            max_train_end_date = end_date

    # print("Min train start date:", min_train_start_date)
    # print("Max train end date:", max_train_end_date)

    min_test_start_date = None
    max_test_end_date = None

    # Iterate through the dataset and get start and end dates
    for entry in dataset.test:
        start_date = entry["start"]
        length_of_series = len(entry["target"])
        frequency = dataset.metadata.freq

        # Calculate the end date
        end_date = start_date + to_offset(frequency) * (length_of_series - 1)

        if min_test_start_date is None or start_date < min_test_start_date:
            min_test_start_date = start_date

        if max_test_end_date is None or end_date > max_test_end_date:
            max_test_end_date = end_date

        # print(f"Start date: {start_date}, End date: {end_date}, Total timepoints: {length_of_series}")

    # print("Min test start date:", min_test_start_date)
    # print("Max test end date:", max_test_end_date)

    # print("Total #Timestamps in train:", len(list(dataset.train)[0]['target']))
    # print("Total #Timestamps in test:", len(list(dataset.test)[0]['target']))

    print(
        "Dataset:",
        dataset_name,
        "Min train start date:",
        min_train_start_date,
        "Max train end date:",
        max_train_end_date,
        "Min test start date:",
        min_test_start_date,
        "Max test end date:",
        max_test_end_date,
        "Total #Timestamps in train:",
        len(list(dataset.train)[0]["target"]),
        "Total #Timestamps in test:",
        len(list(dataset.test)[0]["target"]),
    )

    print("\n")
