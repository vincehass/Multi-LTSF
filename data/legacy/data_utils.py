import random

import numpy as np
from gluonts.dataset.common import ListDataset
from gluonts.dataset.repository.datasets import get_dataset


class CombinedDatasetIterator:
    def __init__(self, datasets, seed, weights):
        self._datasets = [iter(el) for el in datasets]
        self._weights = weights
        self._rng = random.Random(seed)

    def __next__(self):
        (dataset,) = self._rng.choices(self._datasets, weights=self._weights, k=1)
        return next(dataset)


class CombinedDataset:
    def __init__(self, datasets, seed=None, weights=None):
        self._seed = seed
        self._datasets = datasets
        self._weights = weights
        n_datasets = len(datasets)
        if weights is None:
            self._weights = [1 / n_datasets] * n_datasets

    def __iter__(self):
        return CombinedDatasetIterator(self._datasets, self._seed, self._weights)

    def __len__(self):
        return sum([len(ds) for ds in self._datasets])


def create_sliding_window_dataset(name, window_size, dataset_path, is_train=True):
    # Splits each time series into non-overlapping sliding windows
    global_id = 0

    freq = get_dataset(name, path=dataset_path).metadata.freq
    data = ListDataset([], freq=freq)
    dataset = (
        get_dataset(name, path=dataset_path).train
        if is_train
        else get_dataset(name, path=dataset_path).test
    )

    for x in dataset:
        windows = []
        for i in range(0, len(x["target"]), window_size):
            windows.append(
                {
                    "target": x["target"][i : i + window_size],
                    "start": x["start"] + i,
                    "item_id": str(global_id),
                    "feat_static_cat": np.array([0]),
                }
            )
            global_id += 1
        data += ListDataset(windows, freq=freq)
    return data


def create_sliding_window_validation_dataset(
    name, window_size, dataset_path, data_id, is_train=True
):
    # Splits each time series into non-overlapping sliding windows
    global_id = 0

    freq = get_dataset(name, path=dataset_path).metadata.freq
    data = ListDataset([], freq=freq)
    dataset = (
        get_dataset(name, path=dataset_path).train
        if is_train
        else get_dataset(name, path=dataset_path).test
    )

    for k, x in enumerate(dataset):
        windows = []
        for i in range(0, len(x["target"]), window_size):
            # if len(x['target'][i:i+window_size]) < window_size: continue
            windows.append(
                {
                    "target": x["target"][i : i + window_size],
                    "start": x["start"] + i,
                    "item_id": str(global_id),
                    "feat_static_cat": np.array([0]),
                    "data_id": data_id,
                }
            )
            global_id += 1
        data += ListDataset(windows, freq=freq)

    if len(data) == 0:
        print("Dataset:", name, "is_train:", is_train, "has no data")

    return data


def create_test_dataset(name, dataset_path, window_size):
    dataset = get_dataset(name, path=dataset_path)
    freq = dataset.metadata.freq
    prediction_length = dataset.metadata.prediction_length

    data = []
    for x in dataset.test:
        offset = len(x["target"]) - (window_size + prediction_length)
        if offset > 0:
            target = x["target"][-(window_size + prediction_length) :]
            data.append(
                {
                    "target": target,
                    "start": x["start"] + offset,
                }
            )
        else:
            data.append(x)
    return ListDataset(data, freq=freq), prediction_length
