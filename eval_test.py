#%%
import pandas as pd
import numpy as np
#%%
source = "./labeled_train_data_imputed_KNNImputer(add_indicator-False copy-True metric-nan_euclidean missing_values-nan n_neighbors-5 weights-distance).csv"
feature_source = "./selected_features_10_04_2020_14_50_38.csv"
pd_data = pd.read_csv(source, encoding = 'big5')

target = "R1COOH"

"""
Split the labeled training data here 
to choose the best model.
With imputed X
"""
from sklearn.model_selection import train_test_split
train, test = train_test_split(pd_data, test_size=1e-9)

train = pd_data.copy()

"""
Here we lose one data point (test_size can't be zero)
to use sklearn shuffle
so that R2 does not drop significantly.
(There is not really a split here.)
"""
print(train.shape)
#train = labeled_train.copy()
y_train=train.pop(target)
x_train=train
#%%
"""
Model Exploration
"""
from xgboost import XGBRegressor
from sklearn.linear_model import BayesianRidge, Ridge, ElasticNet, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
#%%
"""
Run Feature Selection with the best model here!
"""
features_list = pd.read_csv(feature_source)
#selected_mask = list(features_list.iloc[3].dropna())
selected_mask = list(features_list.iloc[1].dropna())

#%%
print(f"num feat: {len(selected_mask)}")
print(f"{selected_mask}")
#%%
#features = list(x_train.columns)
features = selected_mask
X_train = x_train[selected_mask]
# %%
unlabeled_train = pd_data[pd_data[target].isnull()]
print(f"unlabeled: {unlabeled_train.shape}")

"""
TEST!
"""
print("TEST!")
source = "./labeled_test_data_imputed_KNNImputer(add_indicator-False copy-True metric-nan_euclidean missing_values-nan n_neighbors-5 weights-distance).csv"
test_data = pd.read_csv(source, encoding = 'big5')
labeled_test = test_data.loc[test_data[target].notnull()]
#%%
print(labeled_test[target].shape)
#%%
y_true = labeled_test[target]
"""
Fit Model with training data
"""
final_models = [XGBRegressor(),
    GradientBoostingRegressor(),
    ExtraTreesRegressor()]

final_models = [XGBRegressor(),
    GradientBoostingRegressor(),
    ExtraTreesRegressor()]
#%%
y_pred_models = []
r2s = []
for final_model in final_models:
    final_model.fit(X_train,y_train)
    y_pred = final_model.predict(labeled_test[features])
    y_pred_models.append(y_pred)
    r2_fin = r2_score(y_true, y_pred)
    r2s.append(r2_fin)
    print(f"{final_model.__class__.__name__}: {r2_fin}")
#%%
"""
Predict Y
"""
print(labeled_test[features].shape)
y_pred = final_model.predict(labeled_test[features])
print(y_pred.shape)
# %%
import matplotlib.pyplot as plt
#import uuid
import datetime
plt.figure()
n_y = len(y_pred)
y_mean = [y_true.mean() for x in range(n_y)]
xs = [x for x in range(n_y)]
plt.plot(xs,y_true, label = "True")
plt.plot(xs,y_pred_models[0], label = f"{final_models[0].__class__.__name__}: {r2s[0]}", linestyle='dashed')
plt.plot(xs,y_pred_models[1], label = f"{final_models[1].__class__.__name__}: {r2s[1]}", linestyle='dashed')
plt.plot(xs,y_pred_models[2], label = f"{final_models[2].__class__.__name__}: {r2s[2]}", linestyle='dashed')
plt.plot(xs,y_mean, label = "mean", linestyle='dashed')
plt.title(f'Number of Feature Used: {len(features)}')
plt.xlabel('test samples')
plt.ylabel('target values')
plt.legend()
pic_name = f'y_pred_true_mean_p{len(features)}_{datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")}.jpg'
plt.savefig(pic_name)
print(f"{pic_name} was produced.")
#plt.show()