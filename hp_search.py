import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
from glob import glob
from hashlib import sha1
import json
import argparse
import gc
import torch
from gluonts.evaluation import Evaluator, make_evaluation_predictions
from gluonts.transform import ExpectedNumInstanceSampler
from data.data_utils import (
    CombinedDataset,
    SingleInstanceSampler,
    create_test_dataset,
    create_train_and_val_datasets_with_dates,
)
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    StochasticWeightAveraging,
)
# from lightning.pytorch.callbacks import (
#     EarlyStopping,
#     ModelCheckpoint,
#     StochasticWeightAveraging,
# )
import joblib
import pytorch_lightning as pl

from utils.utils import set_seed
from data.dataset_list import DATASET_NAMES_1 as DATASET_NAMES
from data.test_dataset_list import DATASET_NAMES as TEST_DATASET_NAMES
import optuna
import json
import time
from pathlib import Path
from gluonts.evaluation._base import aggregate_valid
from get_model import get_hp_params, get_hp_params_estimator

class TransformerTuningObjective:  
    def __init__(self, args, metric_type="mean_wQuantileLoss"):
        self.args = args
        self.metric_type = metric_type

    def get_params(self, trial) -> dict:
        return get_hp_params(trial, self.args.model)
     
    def __call__(self, trial):
        params = self.get_params(trial)
        args = self.args
        set_seed(42)
        pl.seed_everything(42)
        # Callbacks
        swa_callbacks = StochasticWeightAveraging(
            swa_lrs=args.swa_lrs,
            swa_epoch_start=args.swa_epoch_start,
            annealing_epochs=args.annealing_epochs,
            annealing_strategy=args.annealing_strategy,
        )
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=0.00,
            patience=int(args.early_stopping_patience),
            verbose=True,
            mode="min",
        )

        # lr_monitor = LearningRateMonitor(logging_interval='step')
        callbacks = [early_stop_callback,
                    # lr_monitor,
                    # model_checkpointing
                    ]
        if args.swa:
            print("Using SWA")
            callbacks.append(swa_callbacks)

        # Create train and test datasets
        dataset_path = Path(args.dataset_path)
        if not args.single_dataset:
            train_dataset_names = args.train_datasets #if not args.debug else TEST_DATASET_NAMES
            for test_dataset in args.test_datasets:
                train_dataset_names.remove(test_dataset)
            print("Training datasets:", train_dataset_names)
            print("Test datasets:", args.test_datasets)
            data_id_to_name_map = {}
            name_to_data_id_map = {}
            for data_id, name in enumerate(train_dataset_names):
                data_id_to_name_map[data_id] = name
                name_to_data_id_map[name] = data_id
            test_data_id = -1
            for name in args.test_datasets:
                data_id_to_name_map[test_data_id] = name
                name_to_data_id_map[name] = test_data_id
                test_data_id -= 1
        else:
            print("Training and test on", args.single_dataset)
            data_id_to_name_map = {}
            name_to_data_id_map = {}
            data_id_to_name_map[0] = args.single_dataset
            name_to_data_id_map[args.single_dataset] = 0

        # Get prediction length and set it if we are in the single dataset
        if args.single_dataset and args.use_dataset_prediction_length:
            _, prediction_length, _ = create_test_dataset(
                args.single_dataset, dataset_path, 0, debug=False
            )
            args.prediction_length = prediction_length

        # Cosine Annealing LR
        if args.use_cosine_annealing_lr:
            cosine_annealing_lr_args = {"T_max": args.cosine_annealing_lr_t_max, \
                                        "eta_min": args.cosine_annealing_lr_eta_min}
        else:
            cosine_annealing_lr_args = {}
        
        train_args = {
            # "ckpt_path":ckpt_path,
            "trainer_kwargs":dict(
                    max_epochs=args.max_epochs,
                    accelerator="gpu",
                    devices=[args.gpu],
                    limit_val_batches=args.limit_val_batches,
                    # logger=logger,
                    callbacks=callbacks,
                    # default_root_dir=fulldir_experiments,
                ),
        }
        if args.model == "lag_llama":
            train_args.update({"data_id_to_name_map":data_id_to_name_map,
            "use_cosine_annealing_lr":args.use_cosine_annealing_lr,
            "cosine_annealing_lr_args":cosine_annealing_lr_args,})
        estimator = get_hp_params_estimator(args, train_args, params)

        # Create samplers
        # Here we make a window slightly bigger so that instance sampler can sample from each window
        # An alternative is to have exact size and use different instance sampler (e.g. ValidationSplitSampler)
        # We change ValidationSplitSampler to add min_past
        history_length = estimator.context_length + max(estimator.lags_seq)
        prediction_length = args.prediction_length
        window_size = history_length + prediction_length
        print(
            "Context length:",
            estimator.context_length,
            "Prediction Length:",
            estimator.prediction_length,
            "max(lags_seq):",
            max(estimator.lags_seq),
            "Therefore, window size:",
            window_size,
        )

        if args.use_single_instance_sampler:
            estimator.train_sampler = SingleInstanceSampler(
                min_past=estimator.context_length + max(estimator.lags_seq),
                min_future=estimator.prediction_length,
            )
            estimator.validation_sampler = SingleInstanceSampler(
                min_past=estimator.context_length + max(estimator.lags_seq),
                min_future=estimator.prediction_length,
            )
        else:
            estimator.train_sampler = ExpectedNumInstanceSampler(
                num_instances=1.0,
                min_past=estimator.context_length + max(estimator.lags_seq),
                min_future=estimator.prediction_length,
            )
            estimator.validation_sampler = ExpectedNumInstanceSampler(
                num_instances=1.0,
                min_past=estimator.context_length + max(estimator.lags_seq),
                min_future=estimator.prediction_length,
            )

        ## Batch size
        batch_size = args.batch_size


        if not args.single_dataset:
            # Create training and validation data
            train_datasets, val_datasets, dataset_num_series = [], [], []
            dataset_train_num_points, dataset_val_num_points = [], []

            for data_id, name in enumerate(train_dataset_names):
                data_id = name_to_data_id_map[name]
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
                    data_id,
                    history_length,
                    prediction_length,
                    num_val_windows=args.num_validation_windows,
                    last_k_percentage=args.single_dataset_last_k_percentage
                )
                print(
                    "Dataset:",
                    name,
                    "Total train points:", total_train_points,
                    "Total val points:", total_val_points,
                    "Max Train End Date:",
                    max_train_end_date,
                    "Total %Train points:",
                    total_train_points * 100 / total_points,
                    "Total %Val points:",
                    total_val_windows * 100 / total_points,
                )
                train_datasets.append(train_dataset)
                val_datasets.append(val_dataset)
                dataset_num_series.append(len(train_dataset))
                dataset_train_num_points.append(total_train_points)
                dataset_val_num_points.append(total_val_points)

            # Add test splits of test data to validation dataset, just for tracking purposes
            test_datasets_num_series = []
            test_datasets_num_points = []
            test_datasets = []

            if args.stratified_sampling:
                if args.stratified_sampling == "series":
                    train_weights = dataset_num_series
                    val_weights = dataset_num_series + test_datasets_num_series # If there is just 1 series (airpassengers or saugeenday) this will fail
                elif args.stratified_sampling == "series_inverse":
                    train_weights = [1/x for x in dataset_num_series]
                    val_weights = [1/x for x in dataset_num_series + test_datasets_num_series] # If there is just 1 series (airpassengers or saugeenday) this will fail
                elif args.stratified_sampling == "timesteps":
                    train_weights = dataset_train_num_points
                    val_weights = dataset_val_num_points + test_datasets_num_points
                elif args.stratified_sampling == "timesteps_inverse":
                    train_weights = [1 / x for x in dataset_train_num_points]
                    val_weights = [1 / x for x in dataset_val_num_points + test_datasets_num_points]
            else:
                train_weights = val_weights = None
                
            train_data = CombinedDataset(train_datasets, weights=train_weights)
            val_data = CombinedDataset(val_datasets+test_datasets, weights=val_weights)
        else:
            (
                train_data,
                val_data,
                total_train_points,
                total_val_points,
                total_val_windows,
                max_train_end_date,
                total_points,
            ) = create_train_and_val_datasets_with_dates(
                args.single_dataset,
                dataset_path,
                0,
                history_length,
                prediction_length,
                num_val_windows=args.num_validation_windows,
                last_k_percentage=args.single_dataset_last_k_percentage
            )
            print(
                "Dataset:",
                args.single_dataset,
                "Total train points:", total_train_points,
                "Total val points:", total_val_points,
                "Max Train End Date:",
                max_train_end_date,
                "Total %Train points:",
                total_train_points * 100 / total_points,
                "Total %Val points:",
                total_val_windows * 100 / total_points,
            )

        # Batch size search since when we scale up, we might not be able to use the same batch size for all models
        if args.search_batch_size:
            estimator.num_batches_per_epoch = 10
            estimator.limit_val_batches = 10
            estimator.trainer_kwargs["max_epochs"] = 1
            estimator.trainer_kwargs["callbacks"] = []
            estimator.trainer_kwargs["logger"] = None
            while batch_size >= 1:
                try:
                    print("Trying batch size:", batch_size)
                    # Train
                    predictor = estimator.train_model(
                        training_data=train_data,
                        validation_data=val_data,
                        shuffle_buffer_length=None,
                        ckpt_path=None,
                    )
                    break
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        gc.collect()
                        torch.cuda.empty_cache()
                        if batch_size == 1:
                            print(
                                "Batch is already at the minimum. Cannot reduce further. Exiting..."
                            )
                            exit(0)
                        else:
                            print("Caught OutOfMemoryError. Reducing batch size...")
                            batch_size //= 2
                            continue
                    else:
                        print(e)
                        exit(1)
            estimator.num_batches_per_epoch = args.num_batches_per_epoch
            estimator.limit_val_batches = args.limit_val_batches
            estimator.trainer_kwargs["max_epochs"] = args.max_epochs
            estimator.trainer_kwargs["callbacks"] = callbacks
            # estimator.trainer_kwargs["logger"] = logger
            if batch_size > 1: batch_size //= 2
            estimator.batch_size = batch_size
            print("\nUsing a batch size of", batch_size, "\n")
                # wandb.config.update({"batch_size": batch_size}, allow_val_change=True)

        # Train
        train_output = estimator.train_model(
            training_data=train_data,
            validation_data=val_data,
            num_workers=args.num_workers,
            ckpt_path=None,
        )
        print("prediction length", estimator.prediction_length)
        # Set checkpoint path before evaluating
        predictor = estimator.create_predictor(
                    estimator.create_transformation(),
                    estimator.create_lightning_module(),
                )
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=val_data, 
            predictor=predictor,
            num_samples=args.num_samples
        )
        forecasts = list(forecast_it)
        # if layer == layers[0]:
        tss = list(ts_it)
        
        evaluator = Evaluator(num_workers=args.num_workers, aggregation_strategy=aggregate_valid)
        agg_metrics, _ = evaluator(iter(tss), iter(forecasts), num_series=len(val_data))
        return agg_metrics[self.metric_type]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument("-d", "--dataset_path", type=str, default="datasets/")
    parser.add_argument('--num_workers', type=int, default=4, help='num_workers')
    parser.add_argument('--model', type=str, default="lag_transformer", choices=["informer", "etsformer","lag_llama", "lag_transformer"])

    parser.add_argument("--prediction_length", type=int, default=1)
        ## Augmentation hyperparameters
    # Augmentation probability
    parser.add_argument("--aug_prob", type=float, default=0)
    parser.add_argument(
        "--data_normalization", default=None, choices=["mean", "std", "robust", "none"]
    )
    # Frequency Masking
    parser.add_argument(
        "--freq_mask_rate", type=float, default=0.1, help="Rate of frequency masking"
    )

    # Frequency Mixing
    parser.add_argument(
        "--freq_mixing_rate", type=float, default=0.1, help="Rate of frequency mixing"
    )

    # Jitter
    parser.add_argument(
        "--jitter_prob",
        type=float,
        default=0,
        help="Probability of applying Jitter augmentation",
    )
    parser.add_argument(
        "--jitter_sigma",
        type=float,
        default=0.03,
        help="Standard deviation for Jitter augmentation",
    )

    # Scaling
    parser.add_argument(
        "--scaling_prob",
        type=float,
        default=0,
        help="Probability of applying Scaling augmentation",
    )
    parser.add_argument(
        "--scaling_sigma",
        type=float,
        default=0.1,
        help="Standard deviation for Scaling augmentation",
    )

    # Rotation
    parser.add_argument(
        "--rotation_prob",
        type=float,
        default=0,
        help="Probability of applying Rotation augmentation",
    )

    # Permutation
    parser.add_argument(
        "--permutation_prob",
        type=float,
        default=0,
        help="Probability of applying Permutation augmentation",
    )
    parser.add_argument(
        "--permutation_max_segments",
        type=int,
        default=5,
        help="Maximum segments for Permutation augmentation",
    )
    parser.add_argument(
        "--permutation_seg_mode",
        type=str,
        default="equal",
        choices=["equal", "random"],
        help="Segment mode for Permutation augmentation",
    )

    # MagnitudeWarp
    parser.add_argument(
        "--magnitude_warp_prob",
        type=float,
        default=0,
        help="Probability of applying MagnitudeWarp augmentation",
    )
    parser.add_argument(
        "--magnitude_warp_sigma",
        type=float,
        default=0.2,
        help="Standard deviation for MagnitudeWarp augmentation",
    )
    parser.add_argument(
        "--magnitude_warp_knot",
        type=int,
        default=4,
        help="Number of knots for MagnitudeWarp augmentation",
    )

    # TimeWarp
    parser.add_argument(
        "--time_warp_prob",
        type=float,
        default=0,
        help="Probability of applying TimeWarp augmentation",
    )
    parser.add_argument(
        "--time_warp_sigma",
        type=float,
        default=0.2,
        help="Standard deviation for TimeWarp augmentation",
    )
    parser.add_argument(
        "--time_warp_knot",
        type=int,
        default=4,
        help="Number of knots for TimeWarp augmentation",
    )

    # WindowSlice
    parser.add_argument(
        "--window_slice_prob",
        type=float,
        default=0,
        help="Probability of applying WindowSlice augmentation",
    )
    parser.add_argument(
        "--window_slice_reduce_ratio",
        type=float,
        default=0.9,
        help="Reduce ratio for WindowSlice augmentation",
    )

    # WindowWarp
    parser.add_argument(
        "--window_warp_prob",
        type=float,
        default=0,
        help="Probability of applying WindowWarp augmentation",
    )
    parser.add_argument(
        "--window_warp_window_ratio",
        type=float,
        default=0.1,
        help="Window ratio for WindowWarp augmentation",
    )
    parser.add_argument(
        "--window_warp_scales",
        nargs="+",
        type=float,
        default=[0.5, 2.0],
        help="Scales for WindowWarp augmentation",
    )

    # Argument to include time-features
    parser.add_argument(
        "--time_feat",
        help="include time features",
        action="store_true",
    )
    # SWA arguments
    parser.add_argument(
        "--swa", action="store_true", help="Using Stochastic Weight Averaging"
    )
    parser.add_argument("--swa_lrs", type=float, default=1e-2)
    parser.add_argument("--swa_epoch_start", type=float, default=0.8)
    parser.add_argument("--annealing_epochs", type=int, default=10)
    parser.add_argument(
        "--annealing_strategy", type=str, default="cos", choices=["cos", "linear"]
    )
    # CosineAnnealingLR
    parser.add_argument("--use_cosine_annealing_lr", action="store_true", default=False)
    parser.add_argument("--cosine_annealing_lr_t_max", type=int, default=10000)
    parser.add_argument("--cosine_annealing_lr_eta_min", type=float, default=1e-2)

    # Distribution output
    parser.add_argument('--distr_output', type=str, default="studentT", choices=["studentT"])

    # Number of validation windows
    parser.add_argument("--num_validation_windows", type=int, default=14)
    # Training arguments
    parser.add_argument("-b", "--batch_size", type=int, default=2)
    parser.add_argument("-m", "--max_epochs", type=int, default=1000)
    parser.add_argument("-n", "--num_batches_per_epoch", type=int, default=100)
    parser.add_argument("--limit_val_batches", type=int)
    parser.add_argument("--early_stopping_patience", default=50)
    parser.add_argument("--dropout", type=float, default=0.0)

    # Single dataset setup: used typically for finetuning
    parser.add_argument("--single_dataset", type=str)
    parser.add_argument("--use_dataset_prediction_length", action="store_true", default=False)
    parser.add_argument("--single_dataset_last_k_percentage", type=float)

    # Search search_batch_size
    parser.add_argument("--search_batch_size", action="store_true", default=False)

    # Training/validation iterator type switching
    parser.add_argument("--use_single_instance_sampler", action="store_true", default=True)
    
    # Training KWARGS
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-8)

    parser.add_argument("--train_datasets", type=str, nargs="+", default=DATASET_NAMES)
    parser.add_argument("-t", "--test_datasets", type=str, nargs="+", default=[])
    parser.add_argument(
        "--stratified_sampling",
        type=str,
        choices=["series", "series_inverse", "timesteps", "timesteps_inverse"],
    )

    # Evaluation arguments
    parser.add_argument("--num_parallel_samples", type=int, default=100)
    parser.add_argument("--num_samples", type=int, default=100)

    parser.add_argument("--gpu", type=int, default=0)

    args = parser.parse_args()

    set_seed(args.seed)
    pl.seed_everything(args.seed)
    start_time = time.time()
    # study = optuna.create_study(direction="minimize")
    # joblib.dump(study, "study.pkl")
    # logger = CSVLogger("logs", name="informer")
    # study = joblib.load("study.pkl")
    print("model name", str(args.model)+'_'+str(args.train_datasets[0])+'.pkl')
    try:
        # Try loading existing study
        study = joblib.load(str(args.model)+'_'+str(args.train_datasets[0])+'.pkl')
    except FileNotFoundError:
        # Create a new study if it doesn't exist
        study = optuna.create_study(study_name=args.model, direction='minimize')
    try:
        study.optimize(TransformerTuningObjective(args, metric_type="mean_wQuantileLoss"), n_trials=20)
    except KeyboardInterrupt:
        pass  # Catch interruption
    joblib.dump(study, str(args.model)+'_'+str(args.train_datasets[0])+'.pkl')

    print("Number of finished trials: {}".format(len(study.trials)))
    trial = study.best_trial
    print("Best trial:")
    # Get the best hyperparameters
    best_params = study.best_params
    best_value = study.best_value

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    print(time.time() - start_time)
    output_file_path = args.model+'_'+str(args.train_datasets[0])+'best_params.json'
    with open(output_file_path, 'w') as output_file:
        json.dump({'best_params': best_params, 'best_value': best_value}, output_file)

    print(f"Best hyperparameters and value saved to {output_file_path}")
