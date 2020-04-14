#%%
import pandas as pd
import numpy as np
source = "./DataAll_ETL.csv"
pd_data1 = pd.read_csv(source, encoding = 'big5')
pd_data1['PdateTime']= pd.to_datetime(pd_data1['PdateTime'],format='%Y/%m/%d %H:%M:%S')

# %%
time_line = '2020/02/16 12:00:00 AM'
time_mask=(pd_data1['PdateTime']<time_line)
pd_data = pd_data1.loc[time_mask].copy()

# %%
print(f"total: {pd_data1.shape}")
print(f"before {time_line} {pd_data.shape}")
# (11970, 176) (11434, 176)
# %%
col_source = "./DataAll_欄位說明_V2.csv"
print(f"load {col_source}")
col_pd_data = pd.read_csv(col_source, encoding = 'big5')


# %%
"""
Put (column_name, is_categorical) into a dictionary
"""
col_cate_dict = dict(zip(col_pd_data.欄位名稱, col_pd_data.is_categorical))
col_cate_dict["SerialNo_SPCData_CP6_14"] = col_cate_dict.pop("SerialNo_Scada_CP6_14")
col_cate_dict["SerialNo_SPCData_CP6_15"] = col_cate_dict.pop("SerialNo_Scada_CP6_15")
col_cate_dict["R2_TEMP_OP_NEXT_1_HR"] = 0

# %%
target = "R1COOH"
y = pd_data.pop(target)
X = pd_data

# %%
"""
Filter Columns by
taking the number of missing values in each column
as threshold (no need to worry about target y).
"""
n = pd_data.shape[0]
na_count = X.isna().sum().to_dict()
na_mask = np.array(list(na_count.values())) < n / 10
#%%
X = X.loc[:,na_mask]

# %%
print(f"dropping columns with data less than {X.shape}, {y.shape}")
# (11434, 150), (11434,)
# %%
print(f"before drop: {X.shape}")
def drop_hand_pick_col(X, columns):
    for col in columns:
        X.drop(col, axis=1, inplace=True)
    return X

col_dropped = ["CreatTime_Scada_CP6_14", "CreatTime_Scada_CP6_15", "CreatTime", "SerialNo", "Z_Capacity_GROUP_ID", "R2_TEMP_OP_NEXT_1_HR"]
X = drop_hand_pick_col(X, col_dropped)
print(f"after dropping: {X.shape}")
# before drop: (11434, 150)
# after drop: (11434, 145)
# %%
from sklearn.preprocessing import OneHotEncoder

#%%
#X_temp = X.copy()
#X = X_temp.copy()
#%%
print(f"before {X.shape}")
enc = OneHotEncoder(handle_unknown='ignore')
# Improvement: could drop single-valued cate feature here
#%%
mean_or_mode = []
cate_table = pd.DataFrame()
enc_col_ord = []
for col in X.columns:
    if col_cate_dict[col] == 1:
        # impute value using mode
        mode_val = X.loc[:,col].mode()[0]
        mean_or_mode.append(mode_val)
        X.fillna({col: mode_val}, inplace=True)
        # record the encode cate col order
        enc_col_ord.append(col)
        cate_table = pd.concat([cate_table, X.loc[:,col]], axis=1) 
        X.drop(col, axis=1, inplace=True)
    if col_cate_dict[col] == 0:
        # impute value using means
        mean_val = X.loc[:,col].mean()
        mean_or_mode.append(mean_val)
        X.fillna({col: mean_val}, inplace=True)
        # could add an normalization option here
#%%
print(f"after removing cate feat {X.shape}")
# 145 -> 139 6 cate
# (11434, 139)
# %%
print(f"cate feat {cate_table.shape}")


# %%
enc.fit(cate_table)
new_cols_names = enc.get_feature_names(enc_col_ord)
print(f"one hot encoder feat {new_cols_names}")
# %%
cate_feat_enc = pd.DataFrame(enc.transform(cate_table).toarray(), columns=new_cols_names)
# %%
# %%
cate_feat_enc.shape
print(f"one hot encoder feat {cate_feat_enc.shape}")
# (11434, 17)
# %%
X.reset_index(inplace=True)
X_enc = pd.concat([X, cate_feat_enc], axis=1, ignore_index=True)
#%%
#X_enc.shape
# %%
y.reset_index(drop=True, inplace=True)
out_dat = pd.concat([X_enc, y], axis=1, ignore_index=True)

#%%
final_col_name = list(X.columns)
final_col_name.extend(new_cols_names.tolist())
final_col_name.append(target)
#%%
out_dat.columns=final_col_name
#%%
out_dat.drop("index", axis=1, inplace=True)

#%%
print(f"training feat + target {out_dat.shape}")
# %%
path = f"./impute_dat_encode_cate_train.csv"
out_dat.to_csv(path, index=False, header=True)
print(path)
# %%

print("TEST START")
time_mask=(pd_data1['PdateTime']>=time_line)
pd_data_test = pd_data1.loc[time_mask].copy()

# %%
print(f"test data: {pd_data_test.shape}")

# %%
y_test = pd_data_test.pop(target)
X_test = pd_data_test.loc[:,na_mask]
X_test = drop_hand_pick_col(X_test, col_dropped)

# %%
print(X_test.shape, y_test.shape)

# %%
cate_table_test = pd.DataFrame()
enc_col_ord_test = []
for j, c in enumerate(X_test.columns):
    X_test.fillna({c: mean_or_mode[j]}, inplace=True)
    if col_cate_dict[c] == 1:

        enc_col_ord_test.append(c)
        cate_table_test = pd.concat([cate_table_test, X_test.loc[:,c]], axis=1) 
        X_test.drop(c, axis=1, inplace=True)        

# %%
#%%
cate_table_test.shape
#%%
print("enc_col_ord")
enc_col_ord
# %%
new_cols_names = enc.get_feature_names(enc_col_ord_test)
cate_feat_enc = pd.DataFrame(enc.transform(cate_table_test).toarray(), columns=new_cols_names)
#%%
X_test.reset_index(inplace=True)
X_test_enc = pd.concat([X_test, cate_feat_enc], axis=1, ignore_index=True)

# %%
print(f"X_test_enc.shape: {X_test_enc.shape}")

# %%
y_test.reset_index(drop=True, inplace=True)
out_dat_test = pd.concat([X_test_enc, y_test], axis=1, ignore_index=True)
out_dat_test.columns=final_col_name
out_dat_test.drop("index", axis=1, inplace=True)
#%%
out_dat_test.shape

path_test = f"./impute_dat_encode_cate_test.csv"
out_dat_test.to_csv(path_test, index=False, header=True)
print(path_test)
# %%
print("out_dat_test.shape")
print(out_dat_test.shape)

# %%
