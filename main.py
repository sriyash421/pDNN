import os
import torch
import argparse
import pytorch_lightning as pl
from utils import read_config, get_early_stopper, get_checkpoint_callback, final_logs, print_dict
from train import Model
from dataset import DatasetModule
import numpy as np
from models.model import pDNN

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config.ini")
parser.add_argument("--num_gpus", type=int, default=0)

if __name__ == "__main__":
    args = parser.parse_args()
    filename = args.config
    gpus = args.num_gpus if args.num_gpus is not None else 0
    params = read_config(filename)
    use_gpu = (args.num_gpus > 0)

    if params["JOB_TYPE"] == "train":
        if not os.path.exists(params["SAVE_DIR"]):
            os.makedirs(params["SAVE_DIR"])
        if not os.path.exists(params["LOG_DIR"]):
            os.makedirs(params["LOG_DIR"])
        if not os.path.exists(params["CHECKPOINTS_DIR"]):
            os.makedirs(params["CHECKPOINTS_DIR"])
        
        type2id = dict(
            zip(params["BKG_LIST"]+params["SIG_LIST"], range(len(params["BKG_LIST"])+len(params["SIG_LIST"]))))
        if params["MISSING_TRAIN"] :
            temp = params["BKG_LIST"]+list(set(params["SIG_LIST"]) | set(params["MISSING_SIG"]))
            type2id = dict(zip(temp, range(len(temp))))
        print_dict(type2id, "Types")
        
        dataset = DatasetModule(root_path=params["ROOT_PATH"],
                                campaigns=params["CAMPAIGN"],
                                channel=params["CHANNEL"],
                                norm_array=params["NORM_ARRAY"],
                                sig_sum=params["SIG_SUM"],
                                bkg_sum=params["BKG_SUM"],
                                bkg_list=params["BKG_LIST"],
                                sig_list=params["SIG_LIST"],
                                data_list=params["DATA_LIST"],
                                selected_features=params["FEATURES"],
                                reset_feature=params["RESET_FEATURE"],
                                reset_feature_name=params["RESET_FEATURE_NAME"],
                                rm_negative_weight_events=params["NEGATIVE_WT"],
                                cut_features=params["CUT_FEATURES"],
                                cut_values=params["CUT_VALUES"],
                                cut_types=params["CUT_TYPES"],
                                test_rate=params["TEST_SPLIT"],
                                val_split=params["VAL_SPLIT"],
                                batch_size=params["BATCH_SIZE"],
                                id_dict=type2id,
                                missing_train=params["MISSING_TRAIN"],
                                missing_sig=params["MISSING_SIG"],
                                use_PCA= params["USE_PCA"],
                                pca_components=params["PCA_COMPONENTS"])

        early_stopping, logger, model_checkpoint = None, None, None
        if params["EARLY_STOP"]:
            early_stopping = get_early_stopper(
                monitor=params["ES_MONITOR"], min_delta=params["ES_DELTA"], patience=params["ES_PATIENCE"], mode=params["ES_MODE"])

        if params["SAVE_TB_LOGS"]:
            logger = pl.loggers.TensorBoardLogger(
                save_dir=params["LOG_DIR"], log_graph=False)

        if params["SAVE_MODEL"]:
            model_checkpoint = get_checkpoint_callback(
                PATH=params["CHECKPOINTS_DIR"], monitor='val_loss', save_last=params["CHECK_EPOCH"])  #

        loss_fn, output_fn = None, None
        if params["LOSS"] == "bce_loss":
            loss_fn = torch.nn.BCELoss()
            output_fn = torch.nn.Sigmoid()
        elif params["LOSS"] == "hinge_loss":
            loss_fn = torch.nn.HingeLoss()
            output_fn = torch.nn.Tanh()

        model = Model(momentum=params["MOMENTUM"],
                      nesterov=params["NESTEROV"],
                      learn_rate=params["LEARN_RATE"],
                      learn_rate_decay=params["LR_DECAY"],
                      sig_class_weight=params["SIG_WT"],
                      bkg_class_weight=params["BKG_WT"],
                      threshold=params["THRESHOLD"],
                      optimizer=params["OPT"],
                      loss_fn=loss_fn,
                      output_fn=output_fn,
                      layers=params["LAYERS"],
                      nodes=params["NODES"],
                      dropout=params["DROPOUT"],
                      activation=params["ACTIVATION"],
                      input_size=params["PCA_COMPONENTS"] if params["USE_PCA"] else len(params["FEATURES"]),
                      id_dict=type2id,
                      save_tb_logs=params["SAVE_TB_LOGS"],
                      save_metrics=params["METRICS"],
                      save_wt_metrics=params["WT_METRICS"])

        trainer = pl.Trainer(early_stop_callback=early_stopping,
                             checkpoint_callback=model_checkpoint,
                             logger=logger,
                             max_epochs=params["EPOCHS"],
                             gpus=gpus)
        '''training the model'''
        trainer.fit(model, dataset)

        test_dataset = dataset.test_dataloader()
        training_metrics = model.metrics
        best_model = model
        if params["ES_RESTORE"]:
            best_model.load_state_dict(torch.load(model_checkpoint.best_model_path)['state_dict'])
        final_logs(best_model.dnn, test_dataset,
                   params["THRESHOLD"], output_fn, type2id, gpus, training_metrics, params["LOG_DIR"])

    elif params["JOB_TYPE"] == "test":
        if not os.path.exists(params["LOAD_DIR"]):
            raise Exception("Model doesnt exist")
        type2id = dict(
            zip(params["BKG_LIST"]+params["SIG_LIST"], range(len(params["BKG_LIST"])+len(params["SIG_LIST"]))))
        print_dict(type2id, "Types")
        dataset = DatasetModule(root_path=params["ROOT_PATH"],
                                campaigns=params["CAMPAIGN"],
                                channel=params["CHANNEL"],
                                norm_array=params["NORM_ARRAY"],
                                sig_sum=params["SIG_SUM"],
                                bkg_sum=params["BKG_SUM"],
                                bkg_list=params["BKG_LIST"],
                                sig_list=params["SIG_LIST"],
                                data_list=params["DATA_LIST"],
                                selected_features=params["FEATURES"],
                                reset_feature=params["RESET_FEATURE"],
                                reset_feature_name=params["RESET_FEATURE_NAME"],
                                rm_negative_weight_events=params["NEGATIVE_WT"],
                                cut_features=params["CUT_FEATURES"],
                                cut_values=params["CUT_VALUES"],
                                cut_types=params["CUT_TYPES"],
                                test_rate=1.0,
                                val_split=0.0,
                                batch_size=params["BATCH_SIZE"],
                                id_dict=type2id)
        early_stopping, logger, model_checkpoint = None, None, None

        loss_fn, output_fn = None, None
        if params["LOSS"] == "bce_loss":
            loss_fn = torch.nn.BCELoss()
            output_fn = torch.nn.Sigmoid()
        elif params["LOSS"] == "hinge_loss":
            loss_fn = torch.nn.HingeLoss()
            output_fn = torch.nn.Tanh()

        model = Model(momentum=params["MOMENTUM"],
                      nesterov=params["NESTEROV"],
                      learn_rate=params["LEARN_RATE"],
                      learn_rate_decay=params["LR_DECAY"],
                      sig_class_weight=params["SIG_WT"],
                      bkg_class_weight=params["BKG_WT"],
                      threshold=params["THRESHOLD"],
                      optimizer=params["OPT"],
                      loss_fn=loss_fn,
                      output_fn=output_fn,
                      layers=params["LAYERS"],
                      nodes=params["NODES"],
                      dropout=params["DROPOUT"],
                      activation=params["ACTIVATION"],
                      input_size=len(params["FEATURES"]),
                      id_dict=type2id,
                      save_tb_logs=params["SAVE_TB_LOGS"],
                      save_metrics=params["METRICS"],
                      save_wt_metrics=params["WT_METRICS"])

        dataset.prepare_data()
        dataset.setup("test")

        test_dataset = dataset.test_dataloader()
        training_metrics = model.metrics
        model.load_state_dict(torch.load(
            params["LOAD_DIR"])['state_dict'], strict=False)
        final_logs(model, test_dataset,
                   params["THRESHOLD"], output_fn, type2id, gpus, None, params["LOG_DIR"])
