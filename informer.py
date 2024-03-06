import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
# import inspect
# import os
# import sys

# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(os.path.dirname(currentdir))
# sys.path.insert(0, parentdir)

import multiprocessing
from itertools import islice
import random
from pathlib import Path
import pandas as pd
import os
# import matplotlib.pyplot as plt
from glob import glob
from hashlib import sha1
import json
import wandb

from gluonts.evaluation import make_evaluation_predictions, Evaluator

from estimator import InformerEstimator
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    StochasticWeightAveraging,
)

from pandas.tseries.frequencies import to_offset
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.dataset.common import ListDataset
from gluonts.time_feature import get_lags_for_frequency
import wandb
import copy
from gluonts.transform import InstanceSampler
import numpy as np
import torch
import argparse
import gc
import pytorch_lightning as pl

import sys
sys.path.append("../research-lag-llama/data")
from dataset_list import DATASET_NAMES_1 as DATASET_NAMES
from data_utils import (
    CombinedDataset,
    SingleInstanceSampler,
    create_test_dataset,
    create_train_and_val_datasets_with_dates,
)
from read_new_dataset import get_ett_dataset, create_train_dataset_without_last_k_timesteps, TrainDatasets, MetaData


def train(args): 
    name = DATASET_NAMES[args.data_id]
    dataset_path = Path(args.dataset_path)
    if args.data_id < 12:
        dataset = get_dataset(name, dataset_path)
        context_length = dataset.metadata.prediction_length*7
        prediction_length = dataset.metadata.prediction_length
    else:
        context_length = 256
        prediction_length = 26

    experiment_name = args.experiment_name+'-'+name
    experiment_id = sha1(experiment_name.encode("utf-8")).hexdigest()[:8]
    fulldir_experiments = os.path.join(args.results_dir, experiment_name, str(args.seed))
    if os.path.exists(fulldir_experiments): print(fulldir_experiments, "already exists.")
    os.makedirs(fulldir_experiments, exist_ok=True)
    checkpoint_dir = os.path.join(fulldir_experiments, "checkpoints")
    # dataset_path = Path(args.dataset_path)
    # Code to retrieve the version with the highest #epoch stored and restore it incl directory and its checkpoint
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    ckpt_path = None
    ckpt_file = checkpoint_dir + "/last.ckpt"
    if os.path.isfile(checkpoint_dir + "/last.ckpt"):
        ckpt_path = ckpt_file

    if ckpt_path: print("Checkpoint", ckpt_path, "retrieved from experiment directory")
    else: print ("No checkpoints found. Training from scratch.")

    logger = WandbLogger(name=experiment_name + "-seed-" + str(args.seed), \
                        save_dir=fulldir_experiments, group=experiment_name, \
                        tags=args.wandb_tags, entity=args.wandb_entity, \
                        project=args.wandb_project, \
                        config=vars(args), id=experiment_id, \
                        allow_val_change=True, mode=args.wandb_mode, settings=wandb.Settings(code_dir="."))
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=int(args.patience), verbose=True, mode="min")
    model_checkpointing = ModelCheckpoint(dirpath=checkpoint_dir, save_last=True, save_top_k=1, \
                                        filename='best-{epoch}-{val_loss:.2f}')
    callbacks = [early_stop_callback,
                model_checkpointing,
                ]


    history_length =  context_length##+ max(lags_seq)
    window_size = history_length + prediction_length

    (
    train_dataset,
    val_dataset,
    total_train_points,
    total_val_points,
    total_val_windows,
    max_train_end_date,
    total_points,
    ) = create_train_and_val_datasets_with_dates(
            name,
            dataset_path,
            args.data_id,
            history_length,
            prediction_length,
            num_val_windows=14,
    )    
    test_data, prediction_length, total_points = create_test_dataset(
        name, dataset_path, window_size,
        )
    estimator = InformerEstimator(
    # freq=freq,
    prediction_length=prediction_length,
    context_length=context_length,
    
    # attention hyper-params
    dim_feedforward=32,
    num_encoder_layers=2,
    num_decoder_layers=2,
    nhead=2,
    activation="relu",
    aug_prob = 1.0,
    # training params
    batch_size=128,
    lr = 1e-4,
    num_batches_per_epoch=100,
    trainer_kwargs=dict(max_epochs=1, accelerator='cpu', logger=logger, callbacks=callbacks,  default_root_dir=fulldir_experiments),
    ckpt_path = ckpt_path,
    time_feat= True)

    predictor = estimator.train(
    training_data=train_dataset,
    validation_data=val_dataset,
    shuffle_buffer_length=1024,
    ckpt_path = ckpt_path)



    forecast_it, ts_it = make_evaluation_predictions(dataset=test_data, predictor=predictor)
    forecasts = list(forecast_it)
    tss = list(ts_it)
    # num_workers is limited to 10 if cpu has more cores
    num_workers = min(multiprocessing.cpu_count(), 10)

    evaluator = Evaluator(num_workers=num_workers)
    agg_metrics, ts_metrics = evaluator(iter(tss), iter(forecasts))
    
    metrics_dir = os.path.join(fulldir_experiments, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    metrics_savepath = metrics_dir + "/" + name + ".json"
    with open(metrics_savepath, "w") as metrics_savefile:
        json.dump(agg_metrics, metrics_savefile)
    
    wandb_metrics = {}
    wandb_metrics["test/" + name + "/" + "CRPS"] = agg_metrics["mean_wQuantileLoss"]
    logger.log_metrics(wandb_metrics)

    wandb.finish()
    print('CRPS on', name, agg_metrics["mean_wQuantileLoss"])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-e", "--experiment_name", type=str, required=True)
    parser.add_argument("-d", "--dataset_path", type=str, default="research-lag-llama/datasets/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_id", type=int, default=0, help="dataset id [0-11]")

    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("-w", "--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="lag-llama-test")
    parser.add_argument('--wandb_tags', nargs='+')
    
    parser.add_argument("--wandb_mode", type=str, default="offline", choices=["offline", "online"])
    parser.add_argument("--limit_val_batches", type=int)

    parser.add_argument("-r", "--results_dir", type=str, required=True)

    parser.add_argument('--patience', type=int, default=50, help='early stopping patience')
    args = parser.parse_args()

    # print args for logging
    for arg in vars(args):
        print(arg,':', getattr(args, arg))
    dataset_name = DATASET_NAMES[args.data_id]
    print('dataset', dataset_name)
    train(args)




