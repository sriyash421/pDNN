import os
import json
from configparser import ConfigParser
import pytorch_lightning as pl
import torch
import numpy as np
from datetime import datetime
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, log_loss, roc_curve
import matplotlib.pyplot as plt


def read_config(filename="config.ini"):
    '''read config'''
    if not os.path.exists(filename):
        raise Exception("Config file not found")
    print(f"Parsing the config file")
    parser = ConfigParser()
    parser.optionxform = str
    parser.read(filename)

    temp = {}
    for section in parser.sections():
        params = parser.items(section)
        for param in params:
            temp[param[0]] = param[1]
    config = dict()

    config["JOB_NAME"] = str(temp["job_name"])
    config["JOB_TYPE"] = str(temp["job_type"])
    if config["JOB_TYPE"] == "train":
        config["SAVE_DIR"] = os.path.join(
            str(temp["save_dir"]), config["JOB_NAME"]+f"_{datetime.now()}".replace(" ", "_"))
        config["LOG_DIR"] = os.path.join(config["SAVE_DIR"], "logs")
        config["CHECKPOINTS_DIR"] = os.path.join(
            config["SAVE_DIR"], "checkpoints")
    elif config["JOB_TYPE"] == "test":
        config["LOAD_DIR"] = str(temp["load_dir"])
        config["RESULTS_DIR"] = str(temp["results_dir"])
        config["LOG_DIR"] = os.path.join(config["RESULTS_DIR"], "logs")

    config["ROOT_PATH"] = str(temp["root_path"])
    config["CAMPAIGN"] = list(json.loads(str(temp["campaigns"])))
    config["CHANNEL"] = str(temp["channel"])
    config["NORM_ARRAY"] = True if temp["norm_array"] == "true" else False
    config["SIG_SUM"] = float(temp["sig_sumofweight"])
    config["BKG_SUM"] = float(temp["bkg_sumofweight"])

    config["BKG_LIST"] = list(json.loads(temp["bkg_list"]))
    config["SIG_LIST"] = list(json.loads(temp["sig_list"]))
    config["DATA_LIST"] = list(json.loads(temp["data_list"]))
    config["FEATURES"] = list(json.loads(temp["selected_features"]))
    config["RESET_FEATURE"] = True if temp["reset_feature"] == "true" else False
    config["RESET_FEATURE_NAME"] = str(temp["reset_feature_name"])
    config["NEGATIVE_WT"] = True if temp["rm_negative_weight_events"] == "true" else False
    config["CUT_FEATURES"] = list(json.loads(str(temp["cut_features"])))
    config["CUT_VALUES"] = list(json.loads(temp["cut_values"]))
    config["CUT_TYPES"] = list(json.loads(str(temp["cut_types"])))
    config["TEST_SPLIT"] = float(temp["test_rate"])
    config["VAL_SPLIT"] = float(temp["val_split"])

    config["LAYERS"] = int(temp["layers"])
    try:
        config["NODES"] = json.loads(temp["nodes"])
    except:
        config["NODES"] = int(temp["nodes"])
    config["DROPOUT"] = float(temp["dropout_rate"])
    config["ACTIVATION"] = str(temp["activation_fn"])
    config["LOSS"] = str(temp["loss_fn"])
    config["OPT"] = str(temp["optimizer"])
    config["MOMENTUM"] = float(temp["momentum"])
    config["NESTEROV"] = True if temp["nesterov"] == "true" else False
    config["LEARN_RATE"] = float(temp["learn_rate"])
    config["LR_DECAY"] = float(temp["learn_rate_decay"])
    config["THRESHOLD"] = float(temp["threshold"])
    config["BATCH_SIZE"] = int(temp["batch_size"])
    config["EPOCHS"] = int(temp["epochs"])
    config["SIG_WT"] = float(temp["sig_class_weight"])
    config["BKG_WT"] = float(temp["bkg_class_weight"])
    config["EARLY_STOP"] = True if temp["use_early_stop"] == "true" else False
    config["ES_MONITOR"] = str(temp["early_stop_monitor"])
    config["ES_DELTA"] = float(temp["early_stop_min_delta"])
    config["ES_PATIENCE"] = int(temp["early_stop_patience"])
    config["ES_MODE"] = str(temp["early_stop_mode"])
    config["ES_RESTORE"] = True if temp["early_stop_restore_best_weights"] == "true" else False

    config["METRICS"] = json.loads(temp["train_metrics"])
    config["WT_METRICS"] = json.loads(temp["train_metrics_weighted"])
    config["SAVE_MODEL"] = True if temp["save_model"] == "true" else False
    config["SAVE_TB_LOGS"] = True if temp["save_tb_logs"] == "true" else False
    config["CHECK_EPOCH"] = True if temp["check_model_epoch"] == "true" else False
    print_dict(config, "Config")
    return config


def print_dict(dict, name):
    '''print dictionaries'''
    print("-"*40+f"{name}"+"-"*40)
    for k, v in dict.items():
        print(f"{k:<50} {v}")


def get_early_stopper(monitor, min_delta, patience, mode):
    print(f"Getting the early stopper")
    early_stopper = pl.callbacks.EarlyStopping(
        monitor=monitor,
        min_delta=min_delta,
        patience=patience,
        mode=mode
    )
    return early_stopper


def get_checkpoint_callback(PATH, monitor, save_last):
    print(f"Getting the checkpoint callback")
    checkpoint = pl.callbacks.ModelCheckpoint(
        filepath=PATH,
        monitor=monitor,
        save_last=save_last
    )
    return checkpoint


def final_logs(model, dataloader, threshold, output_fn, id_dict, use_gpu, training_metrics, log_path):
    '''testing phase'''
    device = torch.device("cuda" if use_gpu else "cpu")

    target = torch.empty(0)
    preds = torch.empty(0)
    scores = torch.empty(0)
    ids = torch.empty(0)

    model = model.to(device)
    model.eval()

    for (i, (features, batch_target, batch_ids)) in enumerate(dataloader):
        batch_scores = model(features.to(device)).detach()
        batch_preds = (output_fn(batch_scores) >= threshold).float().detach()

        scores = torch.cat((scores, batch_scores.cpu()))
        preds = torch.cat((preds, batch_preds.cpu()))
        target = torch.cat((target, batch_target.cpu()))
        ids = torch.cat((ids, batch_ids.cpu()))

    target = np.array(target)
    preds = np.array(preds)
    scores = np.array(scores)
    ids = np.array(ids)
    mc_sig_index = target == 1
    mc_bkg_index = target == 0
    """[summary]
    Here we have
        - target: array of target
        - preds: array of predictions by the model
        - scores: array of scores by the model
        - ids: array of ids. Note: id_dict[{value_of_ids}] = name of sig/bkg
        - training_metric: dictionary containing values of loss and acc during training
        #TODO: Plot required things using the above things
    """

    #plotting score distributon
    false_pos_rate, true_pos_rate, _ = roc_curve(target, scores)
    plt.subplot(2,1,1)
    plt.title("score distribution")
    plt.hist(scores[mc_sig_index], bins=40, label="Signal", range=[0,1], histtype=u"step")
    plt.hist(scores[mc_bkg_index], bins=40, label="Background", range=[0,1], histtype=u"step")
    plt.yscale("log")
    plt.ylabel("#events")
    plt.xlabel("prediction score")
    plt.legend(loc="best")
    #plotting ROC curve
    plt.subplot(2,1,2)
    plt.title("ROC curve")
    plt.plot(false_pos_rate, true_pos_rate)
    plt.xlabel("False Positive Rate"), plt.ylabel("True Positive Rate")
    plt.text(0.8, 0.2, f"AUC = {roc_auc_score(target, scores)}", bbox=dict(facecolor="none", edgecolor="black", boxstyle="square"))
    plt.tight_layout()
    plt.savefig(f"{log_path}/report.png")

    test_metrics = {
        "accuracy" : accuracy_score(target, preds),
        "bce_loss" : log_loss(target, scores),
        "mse_loss" : mean_squared_error(target, scores),
        "auc_roc" : roc_auc_score(target, scores)
    }
    print_dict(test_metrics, "TEST RESULTS")
    return
