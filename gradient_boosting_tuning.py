import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
source = "./labeled_train_data_imputed_KNNImputer(add_indicator-False copy-True metric-nan_euclidean missing_values-nan n_neighbors-5 weights-distance).csv"
feature_source = "./selected_features_10_04_2020_14_50_38.csv"
pd_data = pd.read_csv(source, encoding = 'big5')

target = "R1COOH"
# from sklearn.model_selection import train_test_split
# train, test = train_test_split(labeled_train, test_size=1e-9)

train = pd_data.copy() #

y_train=train.pop(target)
x_train=train



#feature_source = "./selected_features.csv"
features_list = pd.read_csv(feature_source)
selected_mask = list(features_list.iloc[1].dropna())

# selected_mask = ['R5_OUT_TEMP_14', 'R5_IV_14', 'R5_IN_TEMP_15', 'R1_LEVEL',
#        'R1_PRESSURE', 'R2_TEMP', 'R2_LEVEL', 'R4_TEMP', 'R4_AGI_OP',
#        'Z_R1_LEVEL_RT', 'Z_Capacity_GROUP_DualReduce',
#        'Z_SB_PPM_GROUP_titanium', 'Z_SB_PPM_GROUP_ID_2.0']
print(f"p = {len(selected_mask)}")



features = selected_mask
X_train = x_train[selected_mask]

print(X_train.shape)

max_feat = len(features)
model = XGBRegressor()#, max_features=max_feat)
#print(np.arange(0,max_feat, max_feat//5))  
gsc = GridSearchCV(
    estimator=model,
    param_grid={
        'max_depth': range(1, 10, 1),
        #'max_depth': range(1, 15, 1),
        #'n_estimators': np.arange(50,126,25),
        #'max_features': range(0,max_feat, max_feat//5),
        #'min_samples_leaf': range(1,5,1),
    },
    scoring='r2',
    cv=5
)

grid_result = gsc.fit(X_train, y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# for test_mean, train_mean, param in zip(
#         grid_result.cv_results_['mean_test_score'],
#         grid_result.cv_results_['mean_train_score'],
#         grid_result.cv_results_['params']):
#     print("Train: %f // Test : %f with: %r" % (train_mean, test_mean, param))


final_model = XGBRegressor(**grid_result.best_params_)
print(final_model)

# final_model = ExtraTreesRegressor()
final_model.fit(X_train, y_train)


"""
TEST!
"""
print("TEST!")
#source = "./impute_dat_encode_cate_test.csv"
test_source = "labeled_test_data_imputed_KNNImputer(add_indicator-False copy-True metric-nan_euclidean missing_values-nan n_neighbors-5 weights-distance).csv"
test_data = pd.read_csv(test_source, encoding = 'big5')
labeled_test = test_data.loc[test_data[target].notnull()]

#%%
y_true = labeled_test[target]
y_pred = final_model.predict(labeled_test[features])
from sklearn.metrics import r2_score
fin_r2 = r2_score(y_true, y_pred)

import matplotlib.pyplot as plt
#import uuid
import datetime
plt.figure()
n_y = len(y_pred)
y_mean = [y_true.mean() for x in range(n_y)]
xs = [x for x in range(n_y)]
plt.plot(xs,y_true, label = "True")
plt.plot(xs,y_pred, label = f"{final_model.__class__.__name__} {fin_r2}", linestyle='dashed')
plt.plot(xs,y_mean, label = "mean", linestyle='dashed')
plot_title = f'Num of Feats: {len(features)} Best: {grid_result.best_score_} \n using {grid_result.best_params_}'
print(plot_title)
print(f"test r^2 {fin_r2}")
plt.title(plot_title)
plt.xlabel('test samples')
plt.ylabel('target values')
plt.legend()
produced_time = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
pic_name = f'tune{final_model.__class__.__name__}_p{len(features)}_{produced_time}.jpg'
plt.savefig(pic_name)
print(f"{pic_name} was produced.")
