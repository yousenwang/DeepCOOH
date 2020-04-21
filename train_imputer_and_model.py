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
    get_valid_filename, save_object_as_pkl)

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
labeled_train = get_labeled_dat(train_pd, target, verbose=True)

"""
Need to drop columns with no predictive ability (correlation = 0) beforehand
Otherwise, the number of columns outputting will decrease
"""

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer
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

pkl_name = config['imputer_pkl_name']
save_object_as_pkl(imputer, pkl_name)



features = config["selected_features"]
if features == "all":
  train = train_df
else:
  train = train_df[features]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
normed_train_X = pd.DataFrame(scaler.fit_transform(train.to_numpy()), columns=train.columns)

plk_name_stand = 'standardizer.pkl'
save_object_as_pkl(scaler, plk_name_stand)
print(f"{scaler} saved.")


from _keras_model import build_model, PrintDot
p = normed_train_X.shape[1]
model = build_model(p)
model.summary()

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.callbacks import EarlyStopping
# The patience parameter is the amount of epochs to check for improvement
early_stop = EarlyStopping(monitor='val_loss', patience=10)
EPOCHS = 1000

from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks


callbacks = [
  early_stop, 
  PrintDot(),
  # Update the pruning step
  pruning_callbacks.UpdatePruningStep(),
  # Add summaries to keep track of the sparsity in different layers during training
  pruning_callbacks.PruningSummaries(log_dir='/.')]
early_history = model.fit(normed_train_X, train_y, 
                    epochs=EPOCHS, validation_split = 0.2, verbose=0, 
                    callbacks=callbacks)

# model_name = 'DeepCOOH.h5'
# model.save(model_name) 
# print(f"{model_name} saved")
model_name = 'DeepCOOH'
tf.keras.models.save_model(model, model_name)
print(f"{model_name} saved")
