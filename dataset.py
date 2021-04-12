import os
import torch
import uproot
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import pytorch_lightning as pl
from torch.utils.data import random_split
from array_utils import remove_cut_values, remove_negative_weights, norweight, get_tensor
from utils import print_dict
from sklearn.decomposition import PCA 

class DatasetModule(pl.LightningDataModule):

    def __init__(self, root_path,
                 campaigns,
                 channel,
                 norm_array,
                 sig_sum,
                 bkg_sum,
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
                 id_dict,
                 missing_train,
                 missing_sig,
                 use_PCA,
                 pca_components):
        super().__init__()
        self.root_path = root_path
        self.campaigns = campaigns
        self.channel = channel
        self.norm_array = norm_array
        self.sig_sum = sig_sum
        self.bkg_sum = bkg_sum
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
        self.id_dict = id_dict
        self.features_dict = dict(
            zip(self.selected_features, range(len(self.selected_features))))
        print_dict(self.features_dict, "Features")
        self.missing_sig = missing_sig
        self.missing_train = missing_train
        if self.missing_train :
            self.sig_list = [i for i in self.sig_list if i not in missing_sig]
        self.use_PCA = use_PCA
        self.pca_components = pca_components
    
    def read_data(self, class_list, campaign, df, id_df) :
        for class_type in class_list :
            events = uproot.open(
                        f"{self.root_path}/merged/{campaign}/{class_type}.root")
            tree = events[events.keys()[0]]
            features = tree.keys()
            tree_pd = tree.pandas.df(self.selected_features)
            df = pd.concat([df, tree_pd], ignore_index=True)
            id_df = pd.concat([id_df, pd.DataFrame(
                {"id": np.ones(len(tree_pd))*self.id_dict[class_type]})])
        return df, id_df

    def prepare_data(self):
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
        print("Reading Dataset...")
        sig_df = pd.DataFrame()
        bkg_df = pd.DataFrame()

        sig_id = pd.DataFrame()
        bkg_id = pd.DataFrame()
        
        missing_df = pd.DataFrame()
        missing_id = pd.DataFrame()

        for campaign in self.campaigns:
            print(f"Reading campaign: {campaign}...")
            sig_df, sig_id = self.read_data(self.sig_list, campaign, sig_df, sig_id)
            bkg_df, bkg_id = self.read_data(self.bkg_list, campaign, bkg_df, bkg_id)
            if self.missing_train :
                missing_df, missing_id = self.read_data(self.missing_sig, campaign, missing_df, missing_id)
        sig_ones = pd.DataFrame({"target": np.ones(len(sig_df))})
        bkg_zeros = pd.DataFrame({"target": np.zeros(len(bkg_df))})
        missing_ones = pd.DataFrame({"target": np.ones(len(missing_df))})
        if self.use_PCA :
            print("Applying PCA...")
            pca = PCA(n_components=self.pca_components,whiten=True)
            temp_sig = pca.fit_transform(sig_df.to_numpy())
            temp_bkg = pca.fit_transform(bkg_df.to_numpy())
            if self.missing_train :
                temp_missing = pca.fit_transform(missing_df.to_numpy())
            else:
                temp_missing = missing_df.to_numpy()
            print(f"After PCA size: sig:{temp_sig.shape} bkg:{temp_bkg.shape} missing:{temp_missing.shape}")
            self.sig = np.concatenate(
                (temp_sig, sig_id.to_numpy(), sig_ones.to_numpy()), axis=1)
            self.bkg = np.concatenate(
                (temp_bkg, bkg_id.to_numpy(), bkg_zeros.to_numpy()), axis=1)
            self.missing = np.concatenate(
                (temp_missing, missing_id.to_numpy(), missing_ones.to_numpy()), axis=1)
        else :
            self.sig = np.concatenate(
                (sig_df.to_numpy(), sig_id.to_numpy(), sig_ones.to_numpy()), axis=1)
            self.bkg = np.concatenate(
                (bkg_df.to_numpy(), bkg_id.to_numpy(), bkg_zeros.to_numpy()), axis=1)
            self.missing = np.concatenate(
                (missing_df.to_numpy(), missing_id.to_numpy(), missing_ones.to_numpy()), axis=1)

        print(f"No. of signal samples: {self.sig.shape[0]}")
        print(f"No. of background samples: {self.bkg.shape[0]}")
        if self.missing_train :
            print(f"No. of signal samples missing: {self.missing.shape[0]}")
            self.bkg_test_rate = float(self.missing.shape[0])/float(self.sig.shape[0]+self.missing.shape[0])
            self.bkg_val_split = float(self.sig.shape[0]*self.val_split)/float(self.sig.shape[0]+self.missing.shape[0])

    def setup(self, stage):
        '''
        function to create tensordatasets by splitting according to ratio and samplers
        '''
        if not self.use_PCA :
            self.sig = remove_cut_values(
                self.sig, self.cut_features, self.cut_values, self.cut_types, self.features_dict)
            self.bkg = remove_cut_values(
                self.bkg, self.cut_features, self.cut_values, self.cut_types, self.features_dict)
            if self.missing_train :
                self.missing = remove_cut_values(
                    self.missing, self.cut_features, self.cut_values, self.cut_types, self.features_dict)

            if self.rm_negative_weight_events:
                self.sig = remove_negative_weights(self.sig)
                self.bkg = remove_negative_weights(self.bkg)
                if self.missing_train :
                    self.missing = remove_negative_weights(self.missing)

            if self.norm_array:
                self.sig[:, :-2] = norweight(self.sig[:, :-2], self.sig_sum)
                self.bkg[:, :-2] = norweight(self.bkg[:, :-2], self.bkg_sum)
                if self.missing_train :
                    self.missing[:, :-2] = norweight(self.missing[:, :-2], 1000.0)

        # self.data = np.concatenate(
        #     (self.sig, self.bkg), axis=0).astype(dtype=np.float32)

        print(
            f"No. of signal samples after removing features: {self.sig.shape}")
        print(
            f"No. of background samples after removing features: {self.bkg.shape}")

        if self.missing_train :
            print("Check for missing values..")
            assert((np.all(self.missing[:,-2]==self.missing[0,-2]) and np.all(self.sig[:,-2] != self.missing[0,-2])))
            self.test_rate = 0.0
            self.sig_train, self.sig_val, _ = self.split_sets(self.sig)
            self.val_split = 0.0
            print("Splitting for missing signal values..")
            self.sig_test, _, _ = self.split_sets(self.missing)
        else :
            self.sig_train, self.sig_val, self.sig_test = self.split_sets(self.sig)
        if self.missing_train :
            self.test_rate, self.val_split = self.bkg_test_rate, self.bkg_val_split
        self.bkg_train, self.bkg_val, self.bkg_test = self.split_sets(self.bkg)
        self.train = ConcatDataset([self.bkg_train, self.sig_train])
        self.val = ConcatDataset([self.bkg_val, self.sig_val])
        self.test = ConcatDataset([self.bkg_test, self.sig_test])
        # print(
        #     f"Sig sizes: train:{len(self.sig_train)/(self.sig.shape[0]+self.missing.shape[0])} val:{len(self.sig_val)/(self.sig.shape[0]+self.missing.shape[0])} test_size:{len(self.sig_test)/(self.sig.shape[0]+self.missing.shape[0])}")
        # print(
        #     f"Bkg sizes: train:{len(self.bkg_train)/self.bkg.shape[0]} val:{len(self.bkg_val)/self.bkg.shape[0]} test_size:{len(self.bkg_test)/self.bkg.shape[0]}")
        print(
            f"Final sizes: train:{len(self.train)} val:{len(self.val)} test_size:{len(self.test)}")

    def split_sets(self, data):
        data = np.array(data, dtype=np.float32)
        target_tensor = torch.from_numpy(data[:, -1])
        id_tensor = torch.from_numpy(data[:, -2])
        features_tensor = torch.from_numpy(data[:, :-2])
        total_size = features_tensor.shape[0]
        val_size = int(total_size * self.val_split)
        test_size = int(total_size * self.test_rate)
        train_size = total_size - val_size - test_size
        dataset = TensorDataset(features_tensor, target_tensor, id_tensor)
        return random_split(dataset, [train_size, val_size, test_size])

    def train_dataloader(self):
        train = DataLoader(
            self.train, batch_size=self.batch_size, num_workers=8, shuffle=True)
        return train

    def val_dataloader(self):
        val = DataLoader(self.val, batch_size=self.batch_size,
                         num_workers=8, shuffle=False)
        return val

    def test_dataloader(self):
        test = DataLoader(self.test, batch_size=self.batch_size,
                          num_workers=8, shuffle=False)
        return test

# may write tests here.
