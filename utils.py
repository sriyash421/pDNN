import os
import json
from configparser import ConfigParser
import pytorch_lightning as pl

def read_config(filename="config.ini") :
    if not os.exists(filename) :
        raise Exception("Config file not found")

    parser = ConfigParser()
    parser.optionxform = str
    parser.read(filename)

    temp = {}
    for section in parser.sections :
        params = parser.items(section)
        for param in params:
            temp[param[0]] = param[1]
    config=dict()
    
    config["JOB_NAME"] = str(config["job_name"])
    config["JOB_TYPE"] = str(config["job_type"])
    if config["JOB_TYPE"] == "train" :
        config["SAVE_DIR"] = os.path.join(str(config["save_dir"]),config["JOB_NAME"])
        config["LOG_DIR"] = os.path.join(config["SAVE_DIR"],"logs")
        config["CHECKPOINTS_DIR"] = os.path.join(config["SAVE_DIR"],"checkpoints")
    elif config["JOB_TYPE"] == "test" :
        config["LOAD_DIR"] = str(config["load_dir"])
        config["RESULTS_DIR"] = str(config["results_dir"])
        config["LOG_DIR"] = os.path.join(config["RESULTS_DIR"],"logs")
    
    config["ARR_PATH"] = str(temp["arr_path"])
    config["RUN_TYPE"] = str(temp["run_type"])
    config["RUN_TYPE"] = str(temp["channel"])
    config["NORM_ARRAY"] = True if temp["norm_array"]=="true" else False
    
    config["BKG_LIST"] = list(json.loads(config["bkg_list"]))
    config["SIG_LIST"] = list(json.loads(config["sig_list"]))
    config["DATA_LIST"] = list(json.loads(config["data_list"]))
    config["FEATURES"] = list(json.loads(temp["selected_features"]))
    config["RESET_FEATURE"] = True if temp["reset_feature"]=="true" else False
    config["RESET_FEATURE_NAME"] = str(temp["reset_feature_name"])
    config["NEGATIVE_WT"] = True if temp["rm_negative_weight_events"]=="true" else False
    config["CUT_FEATURES"] = str(temp["cut_features"])
    config["CUT_VALUES"] = json.loads(temp["cut_values"])
    config["CUT_TYPES"] = str(temp["cut_types"])
    config["TEST_SPLIT"] = float(temp["test_rate"])
    config["VAL_SPLIT"] = float(temp["val_split"])
    
    config["LAYERS"] = int(temp["layers"])
    config["NODES"] = int(temp["nodes"])
    config["DROPOUT"] = float(temp["dropout_rate"])
    config["ACTIVATION"] = str(temp["activation_fn"])
    config["LOSS"] = str(temp["loss_fn"])
    config["OPT"] = str(temp["optimizer"])
    config["MOMENTUM"] = float(config["momentum"])
    config["NESTEROV"] = True if temp["nesterov"]=="true" else False
    config["LEARN_RATE"] = float(config["learn_rate"])
    config["LR_DECAY"] = float(config["learn_rate_decay"])
    config["THRESHOLD"] = float(config["threshold"])
    config["BATCH_SIZE"] = int(config["batch_size"])
    config["EPOCHS"] = int(config["epochs"])
    config["SIG_WT"] = float(config["sig_class_weight"])
    config["BKG_WT"] = float(config["bkg_class_weight"])
    config["EARLY_STOP"] = True if temp["use_early_stop"]=="true" else False
    config["ES_MONITOR"] = str(temp["early_stop_monitor"])
    config["ES_DELTA"] = float(temp["early_stop_delta"])
    config["ES_PATIENCE"] = int(temp["early_stop_patience"])
    config["ES_MODE"] = str(temp["early_stop_mode"])
    config["ES_RESTORE"] = True if temp["early_stop_restore_best_weights"]=="true" else False
    
    config["METRICS"] = json.loads(config["train_metrics"])
    config["WT_METRICS"] = json.loads(config["train_metrics_weighted"])
    config["SAVE_MODEL"] = True if temp["save_model"]=="true" else False
    config["SAVE_TB_LOGS"] = True if temp["save_tb_logs"]=="true" else False
    config["CHECK_EPOCH"] = True if temp["check_model_epoch"]=="true" else False
    
    for k,v in config.items() :
        print(f"{k}\t{v}")
        
    return config

def get_early_stopper(monitor, min_delta, patience, mode) :
    early_stopper = pl.callbacks.EarlyStopping(
        monitor=monitor,
        min_delta=min_delta,
        patience=patience,
        mode=mode
    )
    return early_stopper

def get_checkpoint_callback(PATH, monitor, save_last) :
    checkpoint = pl.callbacks.ModelCheckpoint(
        filepath=PATH,
        monitor=monitor,
        save_last=save_last
    )
    return checkpoint

#LOGGER functions #TODO
'''
model is just a nn.Module and dataset is DataLoader
use model(data) to get output

training_metrics contains values of already plotted info
use output to plot scores, test-acc, test-loss, ROC curve
'''
def final_logs(model, dataset, training_metrics, log_path) :
    return