"""
Auther: Ethan Wang
Started Date: 04/15/2020
"""

import pandas as pd
import numpy as np
import json
import requests
import io

from _pandashandler import (
    get_labeled_dat, get_unlabeled_dat, split_into_train_test_by, 
    drop_hand_pick_cols, load_is_cate_dict, drop_all_categorical_features,
    get_valid_filename, norm, save_object_as_pkl)


config_source = "./shinkong-cooh-infofab.json"
with open(config_source) as f:
  config = json.load(f)

pkl_name = config['imputer_pkl_name']

imputer = load_object_from_pkl(pkl_name)
print(f"{imputer} loaded")