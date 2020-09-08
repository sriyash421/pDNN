import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.utils.data import random_split


class DatasetModule(pl.LightningDataModule):

    def __init__(root_path,
                 arr_path,
                 run_type,
                 campaigns,
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
                 batch_size)
    self.root_path = root_path
    self.arr_path = arr_path
    self.run_type = run_type
    self.campaigns = campaigns
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

        : store data in a numpy array of size (total_length x (num_features))
        : store target(0/1) in a numpy array of size (total_length x 1)
        : store id similarly

        return:
            sig_df: pandas of all mc signals
            bkg_df: pandas of all mc backgrounds
            sig_ones: a pandas df with "target" columns, containing total number of sig of ones
            bkg_ones: a pandas df with "target" columns, containing total number of bkg of zeros
        '''
        sig_df = pd.DataFrame()
        bkg_df = pd.DataFrame()
        for campaign in self.campaigns:
            for sig in self.sig_list:
                events = uproot.open(f"{self.root_path}/merged/{campaign}/{sig}.root")
                tree = events[events.keys()[0]]
                features = tree.keys()
                tree_pd = tree.pandas.df(self.selected_features)
                sig_df = pd.concat([sig_df,tree_pd],ignore_index=True)
            for bkg in self.bkg_list:
                events = uproot.open(f"{self.root_path}/merged/{campaign}/{bkg}.root")
                tree = events[events.keys()[0]]
                features = tree.keys()
                tree_pd = tree.pandas.df(self.selected_features)
                bkg_df = pd.concat([bkg_df,tree_pd],ignore_index=True)
        sig_ones = pd.DataFrame({"target" : np.ones(len(sig_df))})
        bkg_zeros = pd.DataFrame({"target" : np.zeros(len(bkg_df))})

        return sig_df, bkg_df, sig_ones, bkg_zeros

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
