import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.utils.data import random_split


class DatasetModule(pl.LightningDataModule):

    def __init__(arr_path,
                 run_type,
                 channel,
                 norm_array,
                 bkg_list,
                 sig_list,
                 data_list,
                 selected_features,
                 reset_feature,
                 reset_feature_name,
                 rm_negative_weight_events,
                 cut_features,
                 cut_values,
                 cut_types,
                 test_rate,
                 val_split,
                 batch_size,
                 id_dict)
    self.arr_path = arr_path
    self.run_type = run_type
    self.channel = channel
    self.norm_array = norm_array
    self.bkg_list = bkg_list
    self.sig_list = sig_list
    self.data_list = data_list
    self.selected_features = selected_features
    self.reset_feature = reset_feature
    self.reset_feature_name = reset_feature_name
    self.rm_negative_weight_events = rm_negative_weight_events
    self.cut_features = cut_features
    self.cut_values = cut_values
    self.cut_types = cut_types
    self.test_rate = test_rate
    self.val_split = val_split
    self.batch_size = batch_size

    def prepare_data(self):
        # TODO
        '''
        function to check data directory and read features using the channels and the arrays,
        then preprocess the data

        : store data in a numpy array of size (total_length x (num_features+1)) "1" for mass
        : store target(0/1) in a numpy array of size (total_length x 1)
        : store id similarly
        '''
        return

    def setup(self, stage):
        # TODO
        '''
        function to create tensordatasets by splitting according to ratio and samplers
        '''
        return

    def train_dataloader(self):
        train = DataLoader(self.train, batch_size=self.batch_size)
        return train

    def val_dataloader(self):
        val = DataLoader(self.val, batch_size=self.batch_size)
        return val

    def test_dataloader(self):
        test = DataLoader(self.test, batch_size=self.batch_size)
        return test

# may write tests here.
