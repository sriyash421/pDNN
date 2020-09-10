import numpy as np
import torch


def get_cut_index(array, cut_value, cut_type):
    pass_index = None
    temp_index = get_cut_index_value(array, cut_value, cut_type)
    if pass_index is None:
        pass_index = temp_index
    else:
        pass_index = np.intersect1d(pass_index, temp_index)
    return pass_index


def get_cut_index_value(array, cut_value, cut_type):
    # Make cuts
    if cut_type == "=":
        pass_index = np.argwhere(array == cut_value)
    elif cut_type == ">":
        pass_index = np.argwhere(array > cut_value)
    elif cut_type == "<":
        pass_index = np.argwhere(array < cut_value)
    else:
        raise ValueError("Invalid cut_type specified.")
    return pass_index.flatten()


def remove_cut_values(array, cut_features, cut_values, cut_types, features_dict):
    print(cut_features, cut_values, cut_types)
    if len(cut_features) > 0:
        print(f"Removing cut_values")
        for (feature, cut_value, cut_type) in zip(cut_features, cut_values, cut_types):
            temp_index = get_cut_index(
                array[:, features_dict[feature]], cut_value, cut_type)
            array = array[temp_index.flatten(), :]
    return array


def norweight(weight_array, norm=1000):
    print(f"Normalising the arrays")
    new = weight_array.copy()
    total_weight = np.sum(new)
    frac = norm / total_weight
    new = frac * new
    return new


def remove_negative_weights(array) :
    print(f"Removing negative weights")
    new = array.copy()
    index = np.argwhere(np.sum(array < 0, axis=1) == 0)
    new = new[index.flatten(), :]
    return new

def get_tensor(A,B, data_type) :
    temp = np.concatenate((A,B), axis=0).astype(dtype=data_type)
    return torch.from_numpy(temp)