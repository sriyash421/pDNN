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
from dataset import DatasetModule

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config.ini")
parser.add_argument("--num_gpus", type=int)

if __name__ == "main":
    args = args.parrse_args()
    filename = args.config
    gpus = args.num_gpus if args.num_gpus is not None else 0
    params = read_config(filename)
    if params["JOB_TYPE"] == "train":
        if not os.path.exists(params["SAVE_DIR"]):
            os.makedirs(params["SAVE_DIR"])
        if not os.path.exists(params["LOG_DIR"]):
            os.makedirs(params["LOG_DIR"])
        if not os.path.exists(params["CHECKPOINTS_DIR"]):
            os.makedirs(params["CHECKPOINTS_DIR"])

        type2id = Dict(
            zip(params["BKG_LIST"], range(len(params["BKG_LIST"]))) +
            zip(params["SIG_LIST"], [len(params["BKG_LIST"]]*len(parms["SIG_LIST"])))
        )

        dataset = DatasetModule(arr_path=params["ARR_PATH"],
                                run_type=params["RUN_TYPE"],
                                channel=params["RUN_TYPE"],
                                norm_array=params["NORM_ARRAY"],
                                bkg_list=params["BKG_LIST"],
                                sig_list=params["SIG_LIST"],
                                data_list=params["DATA_LIST"],
                                selected_features=params["FEATURES"],
                                reset_feature=params["RESET_FEATURE"],
                                reset_feature_name=params["RESET_FEATURE_NAME"],
                                rm_negative_weight_events=params["NEGATIVE_WT"],
                                cut_features=params["CUT_FEATURES"],
                                cut_values=params["CUT_TYPES"],
                                cut_types=params["CUT_TYPES"],
                                test_rate=params["TEST_SPLIT"],
                                val_split=params["VAL_SPLIT"],
                                batch_size=params["BATCH_SIZE"],
                                id_dict=type2id)

        early_stopping, logger, model_checkpoint = None, None, None
        if params["EARLY_STOP"]:
            early_stopping = get_early_stopper(monitor=params["ES_MONITOR"], min_delta=params["ES_DELTA"], patience=params["ES_PATIENCE"], mode)

        if params["SAVE_TB_LOGS"]:
            logger = pl.loggers.TensorboardLogger(path=config["LOG_DIR"])

        if params["SAVE_MODEL"]:
            model_checkpoint = get_checkpoint_callback(
                PATH=params["CHECKPOINTS_DIR"], monitor=params["ES_MONITOR"], save_last=config["CHECK_EPOCH"])  # TODO

        model = Model(momentum=params["MOMENTUM"],
                      nesterov=params["NESTEROV"],
                      learn_rate=params["LEARN_RATE"],
                      learn_rate_decay=params["LR_DECAY"],
                      sig_class_weight=params["SIG_WT"],
                      bkg_class_weight=params["BKG_WT"],
                      threshold=params["THRESHOLD"],
                      optimizer=params["OPT"],
                      loss_fn=params["LOSS"],
                      layers=params["LAYERS"],
                      nodes=params["NODES"],
                      dropout=params["DROPOUT"]
                      activation=params["ACTIVATION"],
                      input_size=len(params["FEATURES"])+1,
                      id_dict=type2id,
                      save_tb_logs=params["SAVE_TB_LOGS"],
                      save_metrics=params["METRICS"],
                      save_wt_metric=params["WT_METRICS"])

        trainer = pl.Trainer(early_stop_callback=early_stopping,
                             model_checkpoint_callback=model_checkpoint,
                             logger=logger,
                             max_epochs=params["EPOCHS"])
        trainer.fit(model, dataset, gpus=gpus)

        test_dataset = dataset.test_dataloader()
        training_metrics = model.metrics
        best_model = pl.model.dnn
        if params["ES_RESTORE"] :
            best_model = pl.load_from_checkpoint("")
        final_logs()  # TODO

    else:
        # TODO: jobtype = val
