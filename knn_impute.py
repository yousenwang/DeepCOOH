"""
Auther: Ethan Wang
Started Date: 04/09/2020
"""
import pandas as pd
import numpy as np
from _pandashandler import (
    get_labeled_dat, get_unlabeled_dat, split_into_train_test_by, 
    drop_hand_pick_cols, load_is_cate_dict, drop_all_categorical_features,
    get_valid_filename)
#%%
source = "./DataAll_ETL.csv"
target = 'R1COOH'
split_time_line = '2020/02/16 12:00:00 AM'
col_dropped = [
    "Line",
    "CreatTime_Scada_CP6_14", 
    "CreatTime_Scada_CP6_15",
    "CreatTime_SPCData_CP6_14",
    "CreatTime_SPCData_CP6_15",
    "CreatTime", 
    "SerialNo",
    "SerialNo_SPCData_CP6_14",
    "SerialNo_SPCData_CP6_15",
    "Z_Capacity_GROUP_ID", 
    "R2_TEMP_OP_NEXT_1_HR"
]
DataAll_ELT_pd = pd.read_csv(source, encoding = 'big5')
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

path = f"./labeled_train_data.csv"
labeled_train.to_csv(path, index=False, header=True)
print(path)

"""
Need to drop columns with no predictive ability (correlation = 0) beforehand
Otherwise, the number of columns outputting will decrease
"""

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import (
    SimpleImputer, KNNImputer, IterativeImputer, MissingIndicator)

unimputed_X = labeled_train.copy()

true_train_y = unimputed_X.pop(target)

unimputed_y = true_train_y.to_numpy()

unimputed_X_cols = unimputed_X.columns

path = f"./unimputed_X.csv"
unimputed_X.to_csv(path, index=False, header=True)
print(path)

print(unimputed_X_cols)
print(unimputed_X.shape)
unimputed_X = unimputed_X.to_numpy()
print(len(unimputed_X), len(unimputed_X[0]))

imputers = [
    #SimpleImputer(missing_values=np.nan, strategy='mean'),
    #SimpleImputer(missing_values=np.nan, strategy='median'),
    #SimpleImputer(missing_values=np.nan, strategy='most_frequent'),
    #KNNImputer(weights='uniform'),
    KNNImputer(weights='distance')
]

"""
‘uniform’ : uniform weights. All points in each neighborhood are weighted equally.
‘distance’ : weight points by the inverse of their distance. in this case, closer neighbors of a query point will have a greater influence than neighbors which are further away.
"""

for imputer in imputers:
    
    unique_name = str(imputer.get_params).strip("<bound method BaseEstimator.get_params of")
    unique_name = get_valid_filename(unique_name)

    path = f"./labeled_train_data_imputed_{unique_name}.csv"
    print(path)
    print("\n")
    out_dat = pd.DataFrame(imputer.fit_transform(unimputed_X), columns=unimputed_X_cols)
    out_dat = pd.concat([out_dat, true_train_y], axis=1)
    print(out_dat.shape)
    out_dat.to_csv(path, index=False, header=True)



"""
TEST!
"""
print("TEST!")
def impute_test_set(test_set, target, imputer):
    unimputed_X = test_set.copy()
    true_test_y = unimputed_X.pop(target)
    unimputed_y = true_test_y.to_numpy()
    unimputed_X_cols = unimputed_X.columns
    out_dat = pd.DataFrame(imputer.transform(unimputed_X), columns=unimputed_X_cols)
    out_dat = pd.concat([out_dat, true_test_y], axis=1)
    print(out_dat.shape)
    path = f"./unimputed_test_X.csv"
    unimputed_X.to_csv(path, index=False, header=True)
    return out_dat

labeled_test = get_labeled_dat(test_pd, target, verbose=True)
path = f"./labeled_test_data.csv"
labeled_test.to_csv(path, index=False, header=True)

for imputer in imputers:
    out_dat = impute_test_set(labeled_test, target, imputer=imputer)
    unique_name = str(imputer.get_params).strip("<bound method BaseEstimator.get_params of")
    unique_name = get_valid_filename(unique_name)
    print(unique_name)
    path = f"./labeled_test_data_imputed_{unique_name}.csv"
    print(path)
    print("\n")
    out_dat.to_csv(path, index=False, header=True)

