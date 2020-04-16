"""
Auther: You Sen Wang (Ethan)
Started Date: 04/15/2020
Email: yousenwang@gmail.com
"""

import pandas as pd
import numpy as np
import json
import requests
import io

from _pandashandler import (
    get_labeled_dat, get_unlabeled_dat, split_into_train_test_by, 
    drop_hand_pick_cols, load_is_cate_dict, drop_all_categorical_features,
    get_valid_filename, load_object_from_pkl)


config_source = "./shinkong-cooh-infofab.json"
with open(config_source) as f:
  config = json.load(f)


all_dat_url = config['all_data_url']
source = config['local_source']
target = config['target']
split_time_line = config['split_time_line']
col_dropped = config['hand_pick_cols_drop']

s = requests.get(all_dat_url).content
DataAll_ELT_pd = pd.read_csv(io.StringIO(s.decode("big5")))

DataAll_ELT_pd.replace(0, np.nan, inplace=True)
DataAll_ELT_pd.dropna(how='all', axis=1, inplace=True) # drop columns if its values are all nulls
DataAll_ELT_pd = drop_hand_pick_cols(DataAll_ELT_pd, col_dropped, verbose=True)
"""
Drop categorical features
"""
is_cate_dict = load_is_cate_dict(path_to_csv='./data_stats.csv', column='is_categorical', encoding='big5', verbose=True)
DataAll_ELT_pd = drop_all_categorical_features(DataAll_ELT_pd, is_cate_dict, verbose=True)
"""
Split Train and Test!
"""
DataAll_ELT_pd['PdateTime'] = pd.to_datetime(DataAll_ELT_pd['PdateTime'],format='%Y/%m/%d %H:%M:%S')
train_pd, test_pd = split_into_train_test_by(DataAll_ELT_pd, 'PdateTime', time_line=split_time_line, drop_col=True, verbose=True)

pkl_name = config['imputer_pkl_name']
imputer = load_object_from_pkl(pkl_name)
print(f"{imputer} loaded")

labeled_test = get_labeled_dat(test_pd, target, verbose=True)

def impute_test_set(test_set, target, imputer):
    true_test_y = test_set.pop(target)
    # unimputed_y = true_test_y.to_numpy()
    unimputed_X_cols = test_set.columns
    out_dat = pd.DataFrame(imputer.transform(test_set), columns=unimputed_X_cols)
    return out_dat, true_test_y

test_df, y_true = impute_test_set(labeled_test, target, imputer)

features = config["selected_features"]
if features == "all":
  test = test_df
else:
  test = test_df[features]

plk_name_stand = 'standardizer.pkl'
scaler = load_object_from_pkl(plk_name_stand)
print(f"{scaler} loaded.")
normed_test_X = pd.DataFrame(scaler.transform(test.to_numpy()), columns=test.columns)



import tensorflow as tf
model_name = 'DeepCOOH.h5'
# Recreate the exact same model, including its weights and the optimizer
model = tf.keras.models.load_model(model_name)

# Show the model architecture
model.summary()

y_pred = model.predict(normed_test_X).flatten()
print(y_pred)
from _plotsforml import plot_tuning_graph
plot_tuning_graph(y_true, y_pred, model, features)

