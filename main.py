import os
import torch
import argparse
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.utils.data import random_split
from utils import read_config, get_early_stopper, get_checkpoint_callback, final_logs
from train import Model

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config.ini")
parser.add_argument("--num_gpus",type=int)

if __name__ == "main" :
    args = args.parrse_args()
    filename = args.config
    gpus = args.num_gpus if args.num_gpus is not None else 0
    params = read_config(filename, "job")
    if job_type == "train" :
        SAVE_DIR = os.path.join(params["save_dir"],params["job_name"])
        if not os.path.exists(SAVE_DIR) :
            os.makedirs(SAVE_DIR)
        
        #TODO: create dataset model
        dataset = None
        # TODO: create a dictionary to map sig/bkg name to an id
        early_stopping = get_early_stopper()#TODO
        logger = pl.loggers.TensorboardLogger() #TODO
        model_checkpoint = get_checkpoint_callback()#TODO
        
        model = Model()#TODO
        trainer = pl.Trainer(early_stop_callback=early_stopping,
                             model_checkpoint_callback=model_checkpoint,
                             logger=logger,
                             max_epochs=EPOCHS) #TODO
        trainer.fit(model, dataset, gpus=gpus)
        
        test_dataset = dataset.test_dataloader()
        training_metrics = model.metrics
        best_model = pl.load_from_checkpoint("")
        final_logs()#TODO
    
    else :
        #TODO: jobtype = val