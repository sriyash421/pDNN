import os
import torch
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.utils.data import random_split
from models.model import pDNN


class Model(pl.Lightning):

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
                 layers,
                 nodes,
                 dropout
                 activation,
                 input_size,
                 id_dict,
                 save_tb_logs,
                 save_metrics,
                 save_wt_metrics
                 ):
        self.dnn = pDNN(layers, nodes, dropout, activation, input_size)
        self.momentum = momentum
        self.nesterov = nesterov
        self.learn_rate = learn_rate
        self.learn_rate_decay = learn_rate_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.sig_class_weight = sig_class_weight
        self.bkg_class_weight = bkg_class_weight
        self.threshold = threshold
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.threshold = threshold
        self.id_dict = id_dict
        self.save_metrics = save_metrics
        self.save_wt_metrics = save_wt_metrics
        self.mse = torch.nn.MSELoss()
        self.softmax = torch.nn.Softmax()
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
        self.save_tb_logs = save_tb_logs

    def forward(self, input):
        return self.dnn(input)

    def configure_optimizers(self):
        optimizer = None
        if self.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.dnn.parameters(
            ), lr=self.learn_rate, betas=[self.momentum, 0.999])
        else:
            optimizer = torch.optim.SGD(
                self.dnn.parameters, lr=self.learn_rate, momentum=self.momentum, nesterov=self.nesterov)

        def scheduler_fn(epoch): return 1./(1+epoch*self.learn_rate_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scheduler_fn)
        return optimizer, scheduler

    def training_step(self, batch, batch_idx):
        inputs, target, ids = batch
        preds = self(inputs)
        loss = loss_fn(preds, target)
        result = pl.TrainResult(loss)
        result = log_metrics(result, loss, "train", preds, target)
        return result

    def validation_step(self, batch, batch_idx):
        inputs, target, ids = batch
        preds = self(inputs)
        loss = self.loss_fn(preds, target)
        result = pl.EvalResult(checkpoint_on=loss)
        result = log_metrics(result, loss, "val", preds, target)
        return result

    def log_metrics(self, result, loss, step, preds, target):
        preds = (self.softmax(preds)>=self.threshold).float()
        result.log(f"{step}_loss", loss)
        
        if "mean_squared_error" in self.save_wt_metrics:
            result.log(f"{step}_mse_loss", (((preds-target)**2)*(target *
                                                                 self.sig_class_weight+(1-target)*self.bkg_class_weight)).mean())
        if "plain_accuracy" in self.save_metrics:
            result.log(f"{step}_plain_accuracy",
                       pl.metric.functional.accuracy(pred, target))
        if "accuracy" in self.save_wt_metrics:
            result.log(f"{step}_accuracy", (((preds == target).float())*(target *
                                                                         self.sig_class_weight+(1-target)*self.bkg_class_weight)).mean())
        return result

    # TODO: Add significance as a metric
    def training_epoch_end(self, outputs):
        self.metrics["train_loss"].append(
            np.mean([output["train_loss"] for output in outputs]))
        
        if "mean_squared_error" in self.save_wt_metrics:
            self.metric["train_mse_loss"].append(
                np.mean([output["train_mse_loss"] for output in outputs]))
        
        if "plain_accuracy" in self.save_metrics:
            self.metrics["train_plain_accuracy"].append(
                np.mean([output["train_plain_accuracy"] for output in outputs]))
        
        if "accuracy" in self.save_metrics:
            self.metrics["train_accuracy"].append(
                np.mean([output["train_accuracy"] for output in outputs]))

        #plot only the unweighted metrics
        if self.save_tb_logs:
            self.logger.experiment.add_scalar(
                "avg_train_loss", self.metrics["train_loss"])
            self.logger.experiment.add_scalar(
                "avg_train_accuracy", self.metrics["train_accuracy"])

    def val_epoch_end(self, outputs):
        self.metrics["val_loss"].append(
            np.mean([output["val_loss"] for output in outputs]))
        
        if "mean_squared_error" in self.save_wt_metrics:
            self.metric["val_mse_loss"].append(
                np.mean([output["val_mse_loss"] for output in outputs]))
        
        if "plain_accuracy" in self.save_metrics:
            self.metrics["val_plain_accuracy"].append(
                np.mean([output["val_plain_accuracy"] for output in outputs]))
        
        if "accuracy" in self.save_metrics:
            self.metrics["val_accuracy"].append(
                np.mean([output["val_accuracy"] for output in outputs]))

        #plot only the unweighted metrics
        if self.save_tb_logs:
            self.logger.experiment.add_scalar(
                "avg_val_loss", self.metrics["val_loss"])
            self.logger.experiment.add_scalar(
                "avg_val_accuracy", self.metrics["val_accuracy"])
