import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir)

import pytorch_lightning as pl
import torch
from gluonts.torch.modules.loss import DistributionLoss, NegativeLogLikelihood
from gluonts.torch.util import weighted_average
import random
from gluonts.torch.util import repeat_along_dim, take_last
from gluonts.itertools import prod
from informer.module import InformerModel
# from aug import freq_mask, freq_mix

class InformerLightningModule(pl.LightningModule):
    def __init__(
        self,
        model_kwargs: dict,
        loss: DistributionLoss = NegativeLogLikelihood(),
        lr: float = 1e-3,
        weight_decay: float = 1e-8,
        aug_prob: float = 0.1,
        aug_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = InformerModel(**self.hparams.model_kwargs)
        self.loss = self.hparams.loss
        self.lr = self.hparams.lr
        self.weight_decay = self.hparams.weight_decay
        self.aug_prob = self.hparams.aug_prob
        self.aug_rate = self.hparams.aug_rate
        self.time_feat = self.hparams.model_kwargs["time_feat"]
        # self.context_length = self.hparams.model_kwargs["context_length"]

    def training_step(self, batch, batch_idx: int):
        """Execute training step"""
        # if random.random() < self.aug_prob:
        #     if random.random() < 0.5:
        #         batch["past_target"], batch["future_target"] = freq_mask(
        #             batch["past_target"], batch["future_target"], rate=self.aug_rate
        #         )
        #     else:
        #         batch["past_target"], batch["future_target"] = freq_mix(
        #             batch["past_target"], batch["future_target"], rate=self.aug_rate
        #         )

        train_loss = self(batch)
        self.log("train_loss", train_loss, on_epoch=True, on_step=False, prog_bar=True)
        return train_loss


    def validation_step(self, batch, batch_idx: int):
        """Execute validation step"""
        with torch.inference_mode():
            val_loss = self(batch)
        self.log("val_loss", val_loss, on_epoch=True, on_step=False, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        """Returns the optimizer to use"""
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    def forward(self, batch):
        past_target = batch["past_target"]
        future_target = batch["future_target"]
        past_observed_values = batch["past_observed_values"]
        future_observed_values = batch["future_observed_values"]
        if self.time_feat:
            past_time_feat = batch["past_time_feat"]
            future_time_feat = batch["future_time_feat"]
        else:
            past_time_feat = None
            future_time_feat = None

        transformer_inputs, loc, scale, _ = self.model.create_network_inputs(past_target= past_target,
                                                                            past_observed_values= past_observed_values,
                                                                            past_time_feat =past_time_feat,
                                                                            future_time_feat=future_time_feat,
                                                                            future_target= future_target,)                                                                  
        params = self.model.output_params(transformer_inputs)
        distr = self.model.output_distribution(params, loc=loc, scale=scale)
        
        loss_values = self.loss(distr, future_target)

        if len(self.model.target_shape) == 0:
            loss_weights = future_observed_values
        else:
            loss_weights, _ = future_observed_values.min(dim=-1, keepdim=False)

        return weighted_average(loss_values, weights=loss_weights)


