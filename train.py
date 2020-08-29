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

    def __init__(self, dnn,
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
                 activation,
                 input_size,
                 id_dict,
                 save_tb_logs,
                 ):
        self.dnn = pDNN(layers, nodes, activation, input_size)
        self.momentum = momentum
        self.nesterov = nesterov
        self.learn_rate = learn_rate
        self.learn_rate_decay = learn_rate_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.sig_class_weight = sig_class_weight
        self.bkg_class_weight = bkg_class_weight
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.threshold = threshold
        self.id_dict = id_dict
        self.metrics = {
            "train_accuracy":[],
            "train_loss":[],
            "val_acuracy":[],
            "val_loss":[],
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
        scheduler_fn = lambda epoch: 1./(1+epoch*self.learn_rate_decay)
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

    def log_metrics(self, result, loss, step, preds, target) :
        result.log(f"{step}_loss", loss)
        result.log(f"{step}_accuracy", pl.metric.functional.accuracy(pred, target))
        return result

    #TODO: Add significance as a metric
    def training_epoch_end(self, outputs) :
        self.metrics["train_accuracy"].append(np.mean([output["train_accuracy"] for output in outputs]))
        self.metrics["train_loss"].append(np.mean([output["train_loss"] for output in outputs]))
        if self.save_tb_logs :
            self.logger.experiment.add_scalar("avg_train_loss", self.metrics["train_loss"])
            self.logger.experiment.add_scalar("avg_train_accuracy", self.metrics["train_accuracy"])
            
        
    def val_epoch_end(self, outputs) :
        self.metrics["val_accuracy"].append(np.mean([output["val_accuracy"] for output in outputs]))
        self.metrics["val_loss"].append(np.mean([output["val_loss"] for output in outputs]))
        if self.save_tb_logs :
            self.logger.experiment.add_scalar("avg_val_loss", self.metrics["val_loss"])
            self.logger.experiment.add_scalar("avg_val_accuracy", self.metrics["val_accuracy"])