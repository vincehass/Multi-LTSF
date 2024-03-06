from typing import Any, Dict, Iterable, List, Optional

import torch
from gluonts.core.component import validated
from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import as_stacked_batches
from gluonts.itertools import Cyclic
from gluonts.time_feature import TimeFeature, time_features_from_frequency_str
from gluonts.torch.distributions import DistributionOutput, StudentTOutput
from gluonts.torch.model.estimator import PyTorchLightningEstimator
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.torch.modules.loss import DistributionLoss, NegativeLogLikelihood
from gluonts.transform import (
    AddAgeFeature,
    AddObservedValuesIndicator,
    DummyValueImputation,
   AddTimeFeatures,
    AsNumpyArray,
    Chain,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    RemoveFields,
    SetField,
    TestSplitSampler,
    Transformation,
    ValidationSplitSampler,
    VstackFeatures,
)
from gluonts.transform.sampler import InstanceSampler
from gluonts.time_feature import get_lags_for_frequency
from informer.lightning_module import InformerLightningModule

PREDICTION_INPUT_NAMES = ["past_target", "past_observed_values"]


TRAINING_INPUT_NAMES = PREDICTION_INPUT_NAMES + [
    "future_target",
    "future_observed_values",
]


class InformerEstimator(PyTorchLightningEstimator):
    @validated()
    def __init__(
        self,
        # freq: str,
        prediction_length: int,
        # Informer arguments
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        d_model: int = 64,
        nhead: int = 4,
        input_size: int = 1,
        activation: str = "gelu",
        dropout: float = 0.1,
        attn: str = "prob",
        factor: int = 2,
        distil: bool = True,
        context_length: Optional[int] = None,
        distr_output: str = "studentT",
        loss: DistributionLoss = NegativeLogLikelihood(),
        scaling: Optional[str] = "std",
        lags_seq: list = ["Q", "M", "W", "D", "H", "T", "S"],
        num_parallel_samples: int = 100,
        batch_size: int = 32,
        num_batches_per_epoch: int = 50,
        weight_decay: float = 1e-8,
        lr: float = 1e-3,
        aug_prob: float = 0.1,
        aug_rate: float = 0.1,
        trainer_kwargs: Optional[Dict[str, Any]] = dict(),
        train_sampler: Optional[InstanceSampler] = None,
        validation_sampler: Optional[InstanceSampler] = None,
        ckpt_path: Optional[str] = None,
        time_feat: bool = False,
    ) -> None:
        trainer_kwargs = {
            "max_epochs": 100,
            **trainer_kwargs,
        }
        super().__init__(trainer_kwargs=trainer_kwargs)

        # self.freq = freq
        self.context_length = (
            context_length if context_length is not None else prediction_length
        )
        self.prediction_length = prediction_length
        self.loss = loss

        self.input_size = input_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.activation = activation
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.attn = attn
        self.factor = factor
        self.distil = distil
        self.lr = lr
        self.weight_decay = weight_decay
        self.aug_prob = aug_prob
        self.aug_rate = aug_rate
        if distr_output == "studentT":
            self.distr_output = StudentTOutput()
        # lag_indices = []
        # for freq in lags_seq:
        #     lag_indices.extend(
        #         get_lags_for_frequency(freq_str=freq, num_default_lags=1)
        #     )

        # if len(lag_indices):
        #     self.lags_seq = sorted(set(lag_indices))
        # #     self.lags_seq = [lag_index - 1 for lag_index in self.lags_seq]
        # else:
        #     self.lags_seq = []
        self.lags_seq = sorted(
            list(
                set(
                    get_lags_for_frequency(freq_str="Q", num_default_lags=1)
                    + get_lags_for_frequency(freq_str="M", num_default_lags=1)
                    + get_lags_for_frequency(freq_str="W", num_default_lags=1)
                    + get_lags_for_frequency(freq_str="D", num_default_lags=1)
                    + get_lags_for_frequency(freq_str="H", num_default_lags=1)
                    + get_lags_for_frequency(freq_str="T", num_default_lags=1)
                    + get_lags_for_frequency(freq_str="S", num_default_lags=1)
                )
            )
        )
        self.lags_seq =  [lag - 1 for lag in self.lags_seq]
        self.scaling = scaling
        self.time_feat = time_feat
        self.num_parallel_samples = num_parallel_samples
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch
        self.ckpt_path = ckpt_path
        self.train_sampler = train_sampler or ExpectedNumInstanceSampler(
            num_instances=1.0, min_future=prediction_length
        )
        self.validation_sampler = validation_sampler or ValidationSplitSampler(
            min_future=prediction_length
        )

    def create_transformation(self) -> Transformation:
        if self.time_feat:
            return Chain(
                [
                    AddTimeFeatures(
                        start_field=FieldName.START,
                        target_field=FieldName.TARGET,
                        output_field=FieldName.FEAT_TIME,
                        time_features=time_features_from_frequency_str("S"),
                        pred_length=self.prediction_length,
                    ),
                    # FilterTransformation(lambda x: sum(abs(x[FieldName.TARGET])) > 0),
                    AddObservedValuesIndicator(
                        target_field=FieldName.TARGET,
                        output_field=FieldName.OBSERVED_VALUES,
                        imputation_method=DummyValueImputation(0.0),
                    ),
                ]
            )
        else:
            return Chain(
                [
                    AddObservedValuesIndicator(
                        target_field=FieldName.TARGET,
                        output_field=FieldName.OBSERVED_VALUES,
                        imputation_method=DummyValueImputation(0.0),
                    ),
                ]
            )

    def _create_instance_splitter(self, module: InformerLightningModule, mode: str):
        assert mode in ["training", "validation", "test"]

        instance_sampler = {
            "training": self.train_sampler,
            "validation": self.validation_sampler,
            "test": TestSplitSampler(),
        }[mode]

        return InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=instance_sampler,
            past_length=module.model._past_length,
            future_length=self.prediction_length,
            time_series_fields=[FieldName.FEAT_TIME, FieldName.OBSERVED_VALUES]
            if self.time_feat
            else [FieldName.OBSERVED_VALUES],
            dummy_value=self.distr_output.value_in_support,
        )

    def create_training_data_loader(
        self,
        data: Dataset,
        module: InformerLightningModule,
        shuffle_buffer_length: Optional[int] = None,
        **kwargs,
    ) -> Iterable:
        data = Cyclic(data).stream()
        instances = self._create_instance_splitter(module, "training").apply(
            data, is_train=True
        )
        if self.time_feat:
            return as_stacked_batches(
                instances,
                batch_size=self.batch_size,
                shuffle_buffer_length=shuffle_buffer_length,
                field_names=TRAINING_INPUT_NAMES
                + ["past_time_feat", "future_time_feat", "data_id", "item_id"],
                output_type=torch.tensor,
                num_batches_per_epoch=self.num_batches_per_epoch,
            )

        else:
            return as_stacked_batches(
                instances,
                batch_size=self.batch_size,
                shuffle_buffer_length=shuffle_buffer_length,
                field_names=TRAINING_INPUT_NAMES + ["data_id", "item_id"],
                output_type=torch.tensor,
                num_batches_per_epoch=self.num_batches_per_epoch,
            )


    def create_validation_data_loader(
        self,
        data: Dataset,
        module: InformerLightningModule,
        **kwargs,
    ) -> Iterable:
        instances = self._create_instance_splitter(module, "validation").apply(
            data, is_train=True
        )
        if self.time_feat:
            return as_stacked_batches(
                instances,
                batch_size=self.batch_size,
                field_names=TRAINING_INPUT_NAMES
                + ["past_time_feat", "future_time_feat", "data_id", "item_id"],
                output_type=torch.tensor,
            )
        else:
            return as_stacked_batches(
                instances,
                batch_size=self.batch_size,
                field_names=TRAINING_INPUT_NAMES + ["data_id", "item_id"],
                output_type=torch.tensor,
            )

    def create_predictor(
        self,
        transformation: Transformation,
        module: InformerLightningModule,
    ) -> PyTorchPredictor:
        prediction_splitter = self._create_instance_splitter(module, "test")

        if self.time_feat:
            return PyTorchPredictor(
                input_transform=transformation + prediction_splitter,
                input_names=PREDICTION_INPUT_NAMES
                + ["past_time_feat", "future_time_feat"],
                prediction_net=module.model,
                batch_size=self.batch_size,
                prediction_length=self.prediction_length,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
        else:
            return PyTorchPredictor(
                input_transform=transformation + prediction_splitter,
                input_names=PREDICTION_INPUT_NAMES,
                prediction_net=module.model,
                batch_size=self.batch_size,
                prediction_length=self.prediction_length,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            )

    def create_lightning_module(self) -> InformerLightningModule:
        model_kwargs = {
            "context_length":self.context_length,
            "prediction_length":self.prediction_length,
            "d_model":self.d_model,
            "nhead":self.nhead,
            "num_encoder_layers":self.num_encoder_layers,
            "num_decoder_layers":self.num_decoder_layers,
            "activation":self.activation,
            "dropout":self.dropout,
            "dim_feedforward":self.dim_feedforward,
            "attn":self.attn,
            "factor":self.factor,
            "distil":self.distil,
            # univariate input
            "input_size":self.input_size,
            "distr_output":self.distr_output,
            "lags_seq":self.lags_seq,
            "scaling":self.scaling,
            "num_parallel_samples":self.num_parallel_samples,
            "time_feat": self.time_feat,
        }
        if self.ckpt_path is not None:
            return InformerLightningModule.load_from_checkpoint(
                self.ckpt_path,
                model_kwargs=model_kwargs,
                loss=self.loss,
                lr=self.lr,
                weight_decay=self.weight_decay,
                aug_prob=self.aug_prob,
                aug_rate=self.aug_rate,
            )
        else:
            return InformerLightningModule(
                model_kwargs=model_kwargs,
                loss=self.loss,
                lr=self.lr,
                weight_decay=self.weight_decay,
                # context_length=self.context_length,
                # prediction_length=self.prediction_length,
                aug_prob=self.aug_prob,
                aug_rate=self.aug_rate,
            )