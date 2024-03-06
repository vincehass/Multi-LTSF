import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

import gzip, json
from gluonts.dataset.common import ListDataset, TrainDatasets, MetaData
from pathlib import Path
from gluonts.dataset.repository.datasets import get_dataset
import os

def create_train_dataset_without_last_k_timesteps(
    raw_train_dataset,
    freq,
    k=0
):
    train_data = []
    for i, series in enumerate(raw_train_dataset):
        s_train = series.copy()
        s_train["target"] = s_train["target"][:len(s_train["target"])-k]
        train_data.append(s_train)
    train_data = ListDataset(train_data, freq=freq)
    return train_data

def load_jsonl_gzip_file(file_path):
    with gzip.open(file_path, 'rt') as f:
        return [json.loads(line) for line in f]

def get_ett_dataset(dataset_name, path):
    dataset_path = Path(path) / dataset_name
    metadata_path = dataset_path / 'metadata.json'
    with open(metadata_path, 'r') as f:
        metadata_dict = json.load(f)
        metadata = MetaData(**metadata_dict)
    # Load train and test datasets
    train_data_path = dataset_path / 'train' / 'data.json.gz'
    test_data_path = dataset_path / 'test' / 'data.json.gz'
    # test dataset
    test_data = load_jsonl_gzip_file(test_data_path)
    # Create GluonTS ListDatasets
    test_ds = ListDataset(test_data, freq=metadata.freq)
    train_ds = create_train_dataset_without_last_k_timesteps(test_ds, freq=metadata.freq, k=24)
    return TrainDatasets(metadata=metadata, train=train_ds, test=test_ds)

def get_household_electricity_dataset(dataset_name, path):
    pass
    # See main

if __name__ == "__main__":
    dataset_name = "household_electricity"

    if dataset_name in ("ett_h1", "ett_h2", "ett_m1", "ett_m2"):
        path = "../research-lag-transformer/datasets/ett_datasets"
        ds = get_ett_dataset(dataset_name, path)
    
    if dataset_name in ("cpu_limit_minute", "cpu_usage_minute", \
                        "function_delay_minute", "instances_minute", \
                        "memory_limit_minute", "memory_usage_minute", \
                        "platform_delay_minute", "requests_minute"):
        path = "../research-lag-llama/research-lag-llama/datasets/huawei/" + dataset_name + ".json"
        with open(path, "r") as f: data = json.load(f)
        metadata = MetaData(**data["metadata"])
        train_data = [x for x in data["train"] if type(x["target"][0]) != str]
        test_data = [x for x in data["test"] if type(x["target"][0]) != str]
        train_ds = ListDataset(train_data, freq=metadata.freq)
        test_ds = ListDataset(test_data, freq=metadata.freq)
        ds = TrainDatasets(metadata=metadata, train=train_ds, test=test_ds)

    if dataset_name in ("beijing_pm25", "AirQualityUCI", "beijing_multisite"):
        path = "../research-lag-transformer/datasets//air_quality/" + dataset_name + ".json"
        with open(path, "r") as f:
            data = json.load(f)
        metadata = MetaData(**data["metadata"])
        train_test_data = [x for x in data["data"] if type(x["target"][0]) != str]
        full_dataset = ListDataset(train_test_data, freq=metadata.freq)
        train_ds = create_train_dataset_without_last_k_timesteps(test_ds, freq=metadata.freq, k=24)
        ds = TrainDatasets(metadata=metadata, train=train_ds, test=full_dataset)

    if dataset_name in ("alibaba_cluster_trace_2018"):
        metadata_path = "/cloudops_dataset/cloudops_dataset/" + dataset_name + "/metadata.json"
        with open(metadata_path, "r") as f:
            metadata_json = json.load(f)
        metadata = MetaData(**metadata_json)
        json_gz_path = "/cloudops_dataset/cloudops_dataset/" + dataset_name + "/all/data.json.gz"
        data = load_jsonl_gzip_file(json_gz_path)
        print(data)

    if dataset_name in ("household_electricity"):
        json_path = "../research-lag-transformer/datasets/household_electricity/individual_household_power_full_dataset.json"
        with open(json_path, "r") as f:
            data = json.load(f)
        test_ds = ListDataset(data, freq="T")
        train_ds = create_train_dataset_without_last_k_timesteps(test_ds, freq="T", k=60)
        metadata = MetaData(freq="T", prediction_length="60")
        ds = TrainDatasets(metadata=metadata, train=train_ds, test=test_ds)
        print(len(ds.train))
        print(len(ds.test))
