import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
from pytorch_lightning.loggers import WandbLogger 
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from gluonts.evaluation import Evaluator, make_evaluation_predictions
from gluonts.evaluation._base import aggregate_valid
from gluonts.transform import ValidationSplitSampler
from gluonts.dataset.repository.datasets import get_dataset

from gluonts.torch.distributions import StudentTOutput, DistributionOutput

import argparse
import json 
from hashlib import sha1
import pytorch_lightning as pl
import os 
from pathlib import Path
import wandb


from gluon.pytorch_estimator import NBEATSEstimator
from gluonts.transform import ExpectedNumInstanceSampler
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint
)

from utils.utils import set_seed, print_gpu_stats

# sys.path.insert(0, '/research-lag-llama')
from data.dataset_list import OLD_DATASET_NAMES
from data.data_utils import CombinedDataset, create_train_and_val_datasets_with_dates, create_test_dataset


def train(args):
    # Set seed
    set_seed(args.seed)
    pl.seed_everything(args.seed)

    # Print GPU stats
    print_gpu_stats()

    # Create a directory to store the results in
    # This string is made independent of hyperparameters here, as more hyperparameters / arguments may be added later
    # The name should be created in the calling bash script
    # This way, when that same script is executed again, automatically the model training is resumed from a checkpoint if available
    experiment_name = args.experiment_name
    experiment_id = sha1(experiment_name.encode("utf-8")).hexdigest()[:8]
    fulldir_experiments = os.path.join(args.results_dir, experiment_name, str(args.seed))
    if os.path.exists(fulldir_experiments): print(fulldir_experiments, "already exists.")
    os.makedirs(fulldir_experiments, exist_ok=True)

    # Create directory for checkpoints
    checkpoint_dir = os.path.join(fulldir_experiments, "checkpoints")

    # Code to retrieve the version with the highest #epoch stored and restore it incl directory and its checkpoint
    ckpt_path = None
    ckpt_file = checkpoint_dir + "/last.ckpt"
    if os.path.isfile(checkpoint_dir + "/last.ckpt"):
        ckpt_path = ckpt_file

    if ckpt_path: print("Checkpoint", ckpt_path, "retrieved from experiment directory")
    else: print ("No checkpoints found. Training from scratch.")

    # W&B logging
    logger = WandbLogger(name=experiment_name + "-seed-" + str(args.seed), \
                        save_dir=fulldir_experiments, group=experiment_name, \
                        tags=args.wandb_tags, entity=args.wandb_entity, \
                        project=args.wandb_project, \
                        config=vars(args), id=experiment_id, \
                        allow_val_change=True, mode=args.wandb_mode, settings=wandb.Settings(code_dir="."))

    # Callbacks
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=int(args.early_stopping_patience), verbose=True, mode="min")
    model_checkpointing = ModelCheckpoint(dirpath=checkpoint_dir, save_last=True, save_top_k=1, \
                                          filename='best-{epoch}-{val_loss:.2f}')
    callbacks = [early_stop_callback, model_checkpointing]

    if args.distr_output == "StudentTOutput":
        distr_output = StudentTOutput()

    args.freq = "w" if args.freq is None else args.freq
    args.lags_seq = [1,2,3,4] if len(args.lags_seq) == 0 else args.lags_seq

    # Create train and test datasets
    dataset_path = Path(args.dataset_path)
    dataset = args.dataset

    # Get prediction length
    raw_dataset = get_dataset(dataset, path=dataset_path)
    args.prediction_length = raw_dataset.metadata.prediction_length

    # Create the estimator
    estimator = NBEATSEstimator(
        prediction_length=args.prediction_length,
        context_length=args.context_length,
        input_size=1,
        batch_size=args.batch_size,
        num_block_layers=args.num_block_layers,
        n_embd_per_block=64,
        num_blocks=args.num_blocks,
        max_context_length=2048,
        scaling=args.data_normalization,
        lr=args.lr,
        weight_decay=args.weight_decay,
        
        # others params designed for trainer
        num_batches_per_epoch=args.num_batches_per_epoch,
        num_parallel_samples=args.num_parallel_samples,
        time_feat=args.time_feat,
        data_id_to_name_map=data_id_to_name_map,
        ckpt_path=ckpt_path,
        trainer_kwargs=dict(
            max_epochs=args.max_epochs,
            accelerator="gpu",
            devices=[args.gpu],
            limit_val_batches=args.limit_val_batches,
            logger=logger,
            callbacks=callbacks,
            default_root_dir=fulldir_experiments,
        ),
    )
    # Save the args as config to the directory
    config_filepath = fulldir_experiments + "/args.json"
    with open(config_filepath, "w") as config_savefile:
        json.dump(vars(args), config_savefile, indent=4)

    # Save the number of parameters to the directory for easy retrieval
    num_parameters = sum(p.numel() for p in estimator.create_lightning_module().parameters())
    num_parameters_path = fulldir_experiments + "/num_parameters.txt"
    with open(num_parameters_path, "w") as num_parameters_savefile:
        num_parameters_savefile.write(str(num_parameters))

    # Create samplers
    # Here we make a window slightly bigger so that instance sampler can sample from each window
    # An alternative is to have exact size and use different instance sampler (e.g. ValidationSplitSampler)
    # We change ValidationSplitSampler to add min_past
    history_length = estimator.context_length + max(estimator.lags_seq)
    print(f"The history length is {history_length}")
    window_size = history_length + estimator.prediction_length
    window_size = 10 * window_size
    estimator.validation_sampler = ValidationSplitSampler(
        min_past=estimator.context_length + max(estimator.lags_seq),
        min_future=estimator.prediction_length,
    )


    if args.evaluate_only:
        pass
    else:
        # Create training data

        (
            train_dataset,
            val_dataset,
            total_train_points,
            total_val_points,
            total_val_windows,
            max_train_end_date,
            total_points,
        ) = create_train_and_val_datasets_with_dates(
            dataset,
            dataset_path,
            data_id=0,
            history_length=history_length,
            prediction_length=args.prediction_length,
            num_val_windows=14, # TODO: Don't hardcode this
        )
        train_data, val_data = train_dataset, val_dataset
    
        # printing the information of resulting train_data 
        print(" Number of time series in train_data: ", len(train_data))

        # Train
        train_output = estimator.train_model(
            training_data=train_data,
            validation_data=val_data,
            shuffle_buffer_length=2048,
            ckpt_path=ckpt_path
        )

    # Set checkpoint path before evaluating
    estimator.ckpt_path = train_output.trainer.checkpoint_callback.best_model_path
    # estimator.ckpt_path = None
    print("Using checkpoint:", estimator.ckpt_path, "for evaluation")

    # Make directory to store metrics
    metrics_dir = os.path.join(fulldir_experiments, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    # Evaluate
    print("Evaluating on", dataset)
    test_data, prediction_length, total_points = create_test_dataset(dataset, dataset_path, window_size)

    # Adapt evaluator to new dataset
    estimator.prediction_length = prediction_length if prediction_length is not None else 0
    estimator.batch_size = max(30 // estimator.prediction_length, 1) # Some heuristic for GPU memory (TODO: change)
    predictor = estimator.create_predictor(
        estimator.create_transformation(),
        estimator.create_lightning_module(),
    )
    # Make evaluations
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_data,
        predictor=predictor,
    )
    forecasts = list(forecast_it)
    tss = list(ts_it)

    # Get metrics
    evaluator = Evaluator(num_workers=1, aggregation_strategy=aggregate_valid)
    agg_metrics, _ = evaluator(
        iter(tss), iter(forecasts), num_series=len(test_data)
    )

    # Save metrics
    metrics_savepath = metrics_dir + "/" + dataset + ".json"
    with open(metrics_savepath, "w") as metrics_savefile:
        json.dump(agg_metrics, metrics_savefile)

    # Log metrics. For now only CRPS is logged.
    wandb_metrics = {}
    wandb_metrics["test/" + dataset + "/" + "CRPS"] = agg_metrics["mean_wQuantileLoss"]
    logger.log_metrics(wandb_metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Experiment args
    parser.add_argument("-e", "--experiment_name", type=str, required=True)

    # Data arguments
    parser.add_argument(
        "-d",
        "--dataset_path",
        type=str,
        default="~/scratch/ts/data",
    )
    parser.add_argument("-t", "--dataset", type=str, default="traffic", choices=OLD_DATASET_NAMES)
    parser.add_argument("--stratified_sampling", action="store_true", default=True)

    # Seed
    parser.add_argument("--seed", type=int, default=42)

    # Model hyperparameters
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--prediction_length", type=int, default=1)
    parser.add_argument("--num_blocks", type=int, default=4)
    parser.add_argument("--num_block_layers", type=int, default=6)
    parser.add_argument("--lags_seq", type=List[int], default=[1])
    

    # Data normalization
    parser.add_argument(
        "--data_normalization", default=None, choices=["mean", "std", "robust", "none"]
    )

    
    # Argument to include time-features
    parser.add_argument(
        "--time_feat",
        help="include time features",
        action="store_true",
    )

    # Training arguments
    parser.add_argument("-b", "--batch_size", type=int, default=100)
    parser.add_argument("-m", "--max_epochs", type=int, default=1000)
    parser.add_argument("-n", "--num_batches_per_epoch", type=int, default=100)
    parser.add_argument("--limit_val_batches", type=int)
    parser.add_argument("--early_stopping_patience", default=50)

    # Evaluation arguments
    parser.add_argument("--num_parallel_samples", type=int, default=100)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=1)

    # GPU ID
    parser.add_argument("--gpu", type=int, default=0)

    # Directory to save everything in
    parser.add_argument("-r", "--results_dir", type=str, default="/home/mila/n/nadhir.hassen/foundatioModel_project/research-lag-llama/scratch")

    # Evaluation arguments
    parser.add_argument("--evaluate_on_train_datasets", action="store_true", default=True, help="Also evaluates in the test splits of the training datasets. Takes too long currently after training. Not advisable to use. Added for future compatability.")

    # W&B
    parser.add_argument("-w", "--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="lag-llama-test")
    parser.add_argument("--wandb_tags", nargs="+")
    parser.add_argument(
        "--wandb_mode", type=str, default="online", choices=["offline", "online"]
    )

    # Other arguments
    parser.add_argument(
        "--evaluate_only", action="store_true", help="Only evaluate, do not train"
    )

    parser.add_argument(
        "--use_kv_cache",
        help="KV caching during infernce",
        action="store_true",
        default=True
    )

    # Debug mode: uses only 3 datasets for faster loading
    parser.add_argument("--debug", action="store_true")

    # Training/validation iterator type switching
    parser.add_argument("--use_single_instance_sampler", action="store_true", default=True)

    # Plot forecasts
    parser.add_argument("--plot_test_forecasts", action="store_true", default=True)

    # Search search_batch_size
    parser.add_argument("--search_batch_size", action="store_true", default=False)

    # Number of validation windows
    parser.add_argument("--num_validation_windows", type=int, default=14)

    # Training KWARGS
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-8)

    # Override arguments with a dictionary file with args
    parser.add_argument('--args_from_dict_path', type=str)

    # Evaluation utils
    parser.add_argument("--eval_prefix", type=str)

    # Other arguments
    parser.add_argument('--evaluate_only', action='store_true', help="Only evaluate, do not train")


    # NOTE: Below arguments added for support finetuning
    # Checkpoints args
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--get_ckpt_path_from_experiment_name", type=str)

    # # Single dataset setup: used typically for finetuning
    # parser.add_argument("--single_dataset", type=str)
    # parser.add_argument("--use_dataset_prediction_length", action="store_true", default=False)


    args = parser.parse_args()

    if args.args_from_dict_path:
        with open(args.args_from_dict_path, "r") as read_file: loaded_args = json.load(read_file)
        for key, value in loaded_args.items():
            setattr(args, key, value)

    # print args for logging
    for arg in vars(args):
        print(arg, ":", getattr(args, arg))

    train(args)