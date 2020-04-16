"""
Author: You Sen Wang (Ethan)
Started Date: 04/09/2020
Email: yousenwang@gmail.com
"""

import pandas as pd
import numpy as np
import pickle

def save_object_as_pkl(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object_from_pkl(filename):
    with open(filename, 'rb') as input:
        obj = pickle.load(input)
    return obj

def get_labeled_dat(X_pd, label, verbose=False):
    X_pd = X_pd.loc[X_pd[label].notnull()]
    X_pd.reset_index(inplace=True, drop=True)
    if verbose:
        print(get_labeled_dat.__name__)
        print(X_pd.shape)
    return X_pd

def get_unlabeled_dat(X_pd, label, verbose=False):
    X_pd = X_pd[X_pd[label].isnull()]
    X_pd.reset_index(inplace=True, drop=True)
    if verbose:
        print(get_unlabeled_dat.__name__)
        print(X_pd.shape)
    return X_pd

def split_into_train_test_by(X_pd, time_col ,time_line, drop_col=False, verbose=False):
    time_mask = (X_pd[time_col]<time_line)
    train_pd = X_pd.loc[time_mask]
    test_pd = X_pd.loc[~time_mask]
    train_pd.reset_index(inplace=True, drop=True)
    test_pd.reset_index(inplace=True, drop=True)
    if drop_col:
        train_pd.drop(time_col, axis=1, inplace=True)
        test_pd.drop(time_col, axis=1, inplace=True)
    if verbose:
        print(f"train: {train_pd.shape}, test: {test_pd.shape}")
    return train_pd, test_pd

def drop_hand_pick_cols(X, columns, verbose=False):
    for col in columns:
        X.drop(col, axis=1, inplace=True)
    if verbose:
        print(X.shape)
    return X

def load_is_cate_dict(path_to_csv='./data_stats.csv', column='is_categorical', encoding='big5', verbose=False):
    col_pd_data = pd.read_csv(path_to_csv, encoding = encoding, index_col=0)
    is_cate_dict = dict(zip(col_pd_data.index, col_pd_data.is_categorical))
    cate_feats_list = [key for key in is_cate_dict.keys() if is_cate_dict[key] == 1]
    if (verbose):
        from collections import Counter
        print(f"load_is_cate_dict {Counter(is_cate_dict.values())}")
        print(cate_feats_list)
    return is_cate_dict

def drop_all_categorical_features(X, is_cate_dict, verbose=False):
    count = 0
    for col in X.columns:
        try: 
            if is_cate_dict[col] == 1:
                X.drop(col, axis=1, inplace=True)
                count+=1
        except KeyError:
            print(f"Key Error! can't find {col}.")
            continue
    if verbose:
        print(f"the number of categorical features dropped: {count}")
    return X

def split_cate_and_cont(X, is_cate_dict, verbose=False):
    cate_feats_pd = pd.DataFrame(dtype="category")
    for col in X.columns:
        if is_cate_dict[col] == 1:
            cate_feats_pd = pd.concat([cate_feats_pd, X.loc[:,col]], axis=1)#.apply(str)], axis=1)
            X.drop(col, axis=1, inplace=True)
    if verbose:
        print(X.shape, cate_feats_pd.shape)
    return X, cate_feats_pd


from sklearn.preprocessing import OneHotEncoder
def one_hot_encode_cate_train(cate_table):
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(cate_table)
    new_cols_names = enc.get_feature_names(cate_table.columns)
    cate_feat_enc = pd.DataFrame(enc.transform(cate_table).toarray(), columns=new_cols_names)
    print(f"one hot encoder feat {cate_feat_enc.shape}")
    return cate_feat_enc, enc

from sklearn.preprocessing import LabelEncoder
def label_encode_cate_train(feature_col, ignore_nan = True):
    return
#     feature = feature_col.copy()
#     if ignore_nan:
#         idx = feature.index.notna()
#         print(idx)
#         feature = feature.dropna(inplace = True)
#     print(feature.shape)
#     label_encoder = LabelEncoder()
#     # Fit the label_encoder
#     label_encoder.fit(feature)
#     # Transform the feature
#     res = label_encoder.transform(feature)
#     print(len(res))
#     # for i, not_nan in enumerate(idx):
#     #     if not_nan:
#     #         feature_col.replace(i, res[i], inplace=True)
#     # print(feature_col)
#     return feature_col, label_encoder

def get_valid_filename(filename):
    import string
    import re
    filename = re.sub(' +', " ", filename)
    
    filename.replace("\t", " ")
    filename.replace("\n", " ")

    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    out_str = ""
    for c in filename:
        if c in valid_chars:
            out_str+= c
        else:
            if c == "=":
                out_str += "-"
                continue
            out_str+=""

    return out_str
 
