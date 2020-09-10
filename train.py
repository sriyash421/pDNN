import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.utils.data import random_split
from models.model import pDNN
import numpy as np


class Model(pl.LightningModule):

    def __init__(self,
                 momentum,
                 nesterov,
                 learn_rate,
                 learn_rate_decay,
                 sig_class_weight,
                 bkg_class_weight,
                 threshold,
                 optimizer,
                 loss_fn,
                 output_fn,
                 layers,
                 nodes,
                 dropout,
                 activation,
                 input_size,
                 id_dict,
                 save_tb_logs,
                 save_metrics,
                 save_wt_metrics
                 ):
        '''create a training class'''
        super(pl.LightningModule, self).__init__()
        self.dnn = pDNN(layers, nodes, dropout, activation, input_size)
        self.example_input_array = torch.ones((1, input_size))
        self.momentum = momentum
        self.nesterov = nesterov
        self.learn_rate = learn_rate
        self.learn_rate_decay = learn_rate_decay
        self.sig_class_weight = sig_class_weight
        self.bkg_class_weight = bkg_class_weight
        self.threshold = threshold
        self.optimizer_ = optimizer
        self.loss_fn = loss_fn
        self.threshold = threshold
        self.id_dict = id_dict
        self.save_metrics = save_metrics
        self.save_wt_metrics = save_wt_metrics
        self.mse = torch.nn.MSELoss()
        self.output_fn = output_fn
        self.metrics = {
            "train_plain_accuracy": [],
            "train_accuracy": [],
            "train_loss": [],
            "train_mse_loss": [],
            "val_plain_accuracy": [],
            "val_accuracy": [],
            "val_loss": [],
            "val_mse_loss": [],
        }
        # self.save_tb_logs = save_tb_logs

    def forward(self, input):
        '''get output'''
        return self.dnn(input)

    def configure_optimizers(self):
        '''create optimizer and scheduler'''
        optimizer = None
        if self.optimizer_ == 'adam':
            optimizer = torch.optim.Adam(self.dnn.parameters(
            ), lr=self.learn_rate, betas=[self.momentum, 0.999])
        else:
            optimizer = torch.optim.SGD(
                self.dnn.parameters(), lr=self.learn_rate, momentum=self.momentum, nesterov=self.nesterov)

        def scheduler_fn(epoch): return 1./(1+epoch*self.learn_rate_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scheduler_fn)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        '''executed during training'''
        inputs, target, ids = batch
        preds = self.output_fn(self(inputs).squeeze())
        loss = self.loss_fn(preds, target)
        result = pl.TrainResult(loss)
        result = self.log_metrics(result, loss, "train", preds, target)
        return result

    def validation_step(self, batch, batch_idx):
        '''executed during validation'''
        inputs, target, ids = batch
        preds = self.output_fn(self(inputs).squeeze())
        loss = self.loss_fn(preds, target)
        result = pl.EvalResult(checkpoint_on=loss)
        result = self.log_metrics(result, loss, "val", preds, target)
        return result

    def log_metrics(self, result, loss, step, preds, target):
        '''log metrics per batch'''
        preds = (preds >= self.threshold).float()
        result.log(f"{step}_loss", loss)

        if "mean_squared_error" in self.save_wt_metrics:
            result.log(f"{step}_mse_loss", (((preds-target)**2)*(target *
                                                                 self.sig_class_weight+(1-target)*self.bkg_class_weight)).mean())
        if "plain_accuracy" in self.save_metrics:
            result.log(f"{step}_plain_accuracy",
                       (preds == target).float().mean())
        if "accuracy" in self.save_wt_metrics:
            result.log(f"{step}_accuracy", (((preds == target).float())*(target *
                                                                         self.sig_class_weight+(1-target)*self.bkg_class_weight)).mean())
        return result

    # TODO: Add significance as a metric
    def training_epoch_end(self, outputs):
        '''log metrics across train epoch'''
        self.log = {}

        self.metrics["train_loss"].append(outputs["train_loss"].mean().item())
        self.log["avg_train_loss"] = self.metrics["train_loss"][-1]

        if "mean_squared_error" in self.save_wt_metrics:
            self.metrics["train_mse_loss"].append(
                outputs["train_mse_loss"].mean().item())
            self.log["avg_train_mse_loss"] = self.metrics["train_mse_loss"][-1]

        if "plain_accuracy" in self.save_metrics:
            self.metrics["train_plain_accuracy"].append(
                outputs["train_plain_accuracy"].mean().item())
            self.log["avg_train_plain_accuracy"] = self.metrics["train_plain_accuracy"][-1]

        if "accuracy" in self.save_metrics:
            self.metrics["train_accuracy"].append(
                outputs["train_accuracy"].mean().item())
            self.log["avg_train_accuracy"] = self.metrics["train_accuracy"][-1]

        # plot only the unweighted metrics
        # if self.save_tb_logs:
        #     self.logger.experiment.add_scalar(
        #         "avg_train_loss", self.metrics["train_loss"][-1])
        #     self.logger.experiment.add_scalar(
        #         "avg_train_accuracy", self.metrics["train_mse_loss"][-1])
        return {"log": self.log}

    def val_epoch_end(self, outputs):
        '''log metrics across val epoch'''
        self.log = {}

        self.metrics["val_loss"].append(outputs["val_loss"].mean().item())
        self.log["avg_val_loss"] = self.metrics["val_loss"][-1]

        if "mean_squared_error" in self.save_wt_metrics:
            self.metrics["val_mse_loss"].append(
                outputs["val_mse_loss"].mean().item())
            self.log["avg_val_mse_loss"] = self.metrics["val_mse_loss"][-1]

        if "plain_accuracy" in self.save_metrics:
            self.metrics["val_plain_accuracy"].append(
                outputs["val_plain_accuracy"].mean().item())
            self.log["avg_val_plain_accuracy"] = self.metrics["val_plain_accuracy"][-1]

        if "accuracy" in self.save_metrics:
            self.metrics["val_accuracy"].append(
                outputs["val_accuracy"].mean().item())
            self.log["avg_val_accuracy"] = self.metrics["val_accuracy"][-1]

        # plot only the unweighted metrics
        # if self.save_tb_logs:
        #     self.logger.experiment.add_scalar(
        #         "avg_val_loss", self.metrics["val_loss"][-1])
        #     self.logger.experiment.add_scalar(
        #         "avg_val_accuracy", self.metrics["val_plain_accuracy"][-1])
        return {"log": self.log}
