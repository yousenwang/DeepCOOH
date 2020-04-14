#%%
import pandas as pd
import numpy as np

#%%
source = "./DataAll_ETL.csv"
DataAll_ELT_pd = pd.read_csv(source, encoding = 'big5')
target = 'R1COOH'
#DataAll_ELT_pd['PdateTime']= pd.to_datetime(DataAll_ELT_pd['PdateTime'],format='%Y/%m/%d %H:%M:%S')
#%%
n = DataAll_ELT_pd.shape[0]
# %%
num_of_zeros = (DataAll_ELT_pd == 0).astype(int).sum(axis=0)
#%%
num_of_nan = DataAll_ELT_pd.isna().sum()
#%%
col_names = ['num_of_zeros', 'num_of_nulls', 'sum_zeros_nulls']
out_dat = pd.concat([num_of_zeros, num_of_nan, num_of_zeros + num_of_nan], axis=1)
out_dat.columns=col_names

#%%
out_dat['zero_null_percent'] = (((num_of_zeros + num_of_nan)/n).round(4))*100

# %%
num_of_zeros.index[num_of_zeros > 0]

# %%
col_source = "./DataAll_欄位說明_V2.csv"
col_pd_data = pd.read_csv(col_source, encoding = 'big5')
col_cate_dict = dict(zip(col_pd_data.欄位名稱, col_pd_data.is_categorical))
col_cate_dict["SerialNo_SPCData_CP6_14"] = col_cate_dict.pop("SerialNo_Scada_CP6_14")
col_cate_dict["SerialNo_SPCData_CP6_15"] = col_cate_dict.pop("SerialNo_Scada_CP6_15")
col_cate_dict["R2_TEMP_OP_NEXT_1_HR"] = 0
#%%
len(col_cate_dict)

# %%
out_dat['is_categorical'] = col_cate_dict.values()
#%%
for key in col_cate_dict.keys():
    out_dat.at[key, 'is_categorical'] = col_cate_dict[key]

#%%
for col in DataAll_ELT_pd.columns:
    if col_cate_dict[col] == 1:
        stats = pd.Series(DataAll_ELT_pd[col], dtype="category").describe(include=['category'])
    else:
        stats = DataAll_ELT_pd[col].describe(include='all')
    for stat, value in zip(stats.index, stats):
        out_dat.at[col, stat] = value
# %%
path = f"./data_stats.csv"
out_dat.to_csv(path, index=True, header=True)
print(path)

# %%
"""
TARGET INFO
"""
from _pandashandler import (get_labeled_dat)
from _plotsforml import (plot_norm, plot_QQ, data_correlation)

target = 'R1COOH'
labeled_dat = get_labeled_dat(DataAll_ELT_pd, target, verbose=True)


plot_norm(labeled_dat[target])
plot_QQ(labeled_dat[target])

log_feat = f"log_of_one_plus_{target}"
labeled_dat[log_feat] = np.log1p(labeled_dat[target])

plot_norm(labeled_dat[log_feat])
plot_QQ(labeled_dat[log_feat])

data_correlation(DataAll_ELT_pd)
