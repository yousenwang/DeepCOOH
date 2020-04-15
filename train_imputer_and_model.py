"""
Auther: Ethan Wang
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
    get_valid_filename, norm, save_object_as_pkl)

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
train_pd, test_pd = split_into_train_test_by(DataAll_ELT_pd, 'PdateTime', time_line=split_time_line, drop_col=True)
labeled_train = get_labeled_dat(train_pd, target, verbose=True)

"""
Need to drop columns with no predictive ability (correlation = 0) beforehand
Otherwise, the number of columns outputting will decrease
"""

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer
# from sklearn.impute import (
#     SimpleImputer, KNNImputer, IterativeImputer, MissingIndicator)

unimputed_X = labeled_train.copy()
train_y = unimputed_X.pop(target)

#unimputed_y = train_y.to_numpy()
unimputed_X_cols = unimputed_X.columns
unimputed_X = unimputed_X.to_numpy()

imputer = KNNImputer(weights='distance')


"""
‘uniform’ : uniform weights. All points in each neighborhood are weighted equally.
‘distance’ : weight points by the inverse of their distance. in this case, closer neighbors of a query point will have a greater influence than neighbors which are further away.
"""

unique_name = str(imputer.get_params).strip("<bound method BaseEstimator.get_params of")
unique_name = get_valid_filename(unique_name)

train_df = pd.DataFrame(imputer.fit_transform(unimputed_X), columns=unimputed_X_cols)
#out_dat = pd.concat([out_dat_X, true_train_y], axis=1)

pkl_name = config['imputer_pkl_name']
save_object_as_pkl(imputer, pkl_name)


features = config["selected_features"]

train = train_df[features]

train_stats =   train.describe()
train_stats = train_stats.transpose()
path = f"./train_stats.csv"
train_stats.to_csv(path, index=True, header=True)

normed_train_X = norm(train, train_stats)

from _keras_model import build_model, PrintDot

p = normed_train_X.shape[1]
model = build_model(p)

model.summary()


import keras

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
EPOCHS = 1000
early_history = model.fit(normed_train_X, train_y, 
                    epochs=EPOCHS, validation_split = 0.2, verbose=0, 
                    callbacks=[early_stop, PrintDot()])

model_name = 'DeepCOOH.h5'
model.save(model_name) 
print(f"{model_name} saved")