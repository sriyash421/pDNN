from configparser import ConfigParser
import pytorch_lightning as pl

def read_config(filename="config.ini",section="") :
    #TODO: update to parse all sections
    if not section:
        raise Exception("Section not specified")

    parser = ConfigParser()
    parser.optionxform = str
    parser.read(filename)

    config = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            config[param[0]] = param[1]
    else:
        raise Exception(
            "Section {0} not found in the {1} file".format(section, filename)
        )
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