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
            "train_history_acc": [],
            "train_history_loss": [],
            "val_history_acc": [],
            "val_history_loss": [],
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

    def step_helper(self, batch):
        inputs, target, ids = batch
        preds = self.output_fn(self(inputs).squeeze())
        loss = self.loss_fn(preds, target)
        preds = (preds >= self.threshold).float()
        accuracy = (preds == target).float().mean()
        return loss, accuracy

    def training_step(self, batch, batch_idx):
        '''executed during training'''
        train_loss, train_acc = self.step_helper(batch)
        return {'loss': train_loss, 'acc': train_acc}

    def validation_step(self, batch, batch_idx):
        '''executed during validation'''
        val_loss, val_acc = self.step_helper(batch)
        return {'val_loss': val_loss, 'val_acc': val_acc}

    # TODO: Add significance as a metric
    def training_epoch_end(self, outputs):
        '''log metrics across train epoch'''
        avg_train_loss = torch.stack([output['loss']
                                      for output in outputs]).mean()
        avg_train_acc = torch.stack([output['acc']
                                     for output in outputs]).mean()
        train_metrics = {'train_loss': avg_train_loss,
                         'train_acc': avg_train_acc}
        self.metrics['train_history_loss'].append(avg_train_loss)
        self.metrics['train_history_acc'].append(avg_train_acc)
        return {
            'progress_bar': train_metrics,
            'log': train_metrics
        }

    def validation_epoch_end(self, outputs):
        '''log metrics across val epoch'''
        avg_val_loss = torch.stack([output['val_loss']
                                    for output in outputs]).mean()
        avg_val_acc = torch.stack([output['val_acc']
                                   for output in outputs]).mean()
        val_metrics = {'val_loss': avg_val_loss, 'val_acc': avg_val_acc}
        self.metrics['val_history_loss'].append(avg_val_loss)
        self.metrics['val_history_acc'].append(avg_val_acc)
        return {
            'progress_bar': val_metrics,
            'log': val_metrics
        }
