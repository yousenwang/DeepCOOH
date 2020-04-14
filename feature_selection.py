"""
Author: You Sen Wang (Ethan)
Date: 04/10/2020
Press Ctrl+End to go to the bottom
"""
import os, sys, click
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from sklearn.ensemble import GradientBoostingRegressor
from _pandashandler import (
    get_labeled_dat)

time_produced = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
def plot_feature_importances(features, feature_importances):
    plt.figure(figsize=(19.2 * 2, 10.8 / 1.3 * 3))
    plt.yscale('log', nonposy='clip')
    #plt.tight_layout()
    plt.bar(range(len(feature_importances)), feature_importances, align='center')
    plt.xticks(range(len(feature_importances)), features, rotation='vertical')
    plt.title('Feature importance')
    plt.ylabel('Importance score')
    plt.xlabel('Features')
    pic_name = f'feature_importance_{time_produced}.jpg'
    plt.savefig(pic_name)
    print(f"{pic_name} was produced.")

def feature_importances_to_csv(features, feature_importances):
    feature_importances = pd.DataFrame(feature_importances, columns=['weights'], index=features)
    feature_importances.sort_values(by=['weights'], ascending=False, inplace=True)
    path = f"./feature_importances_{time_produced}.csv"
    print(f"path: {path}")
    feature_importances.to_csv(path, index=True, header=True)
    return feature_importances

def get_thresholds(N=9, verbose=False):
    # start = feature_importances["weights"].min()
    # stop = feature_importances["weights"].max()
    # thresholds = np.linspace(start, stop, num=100)

    thresholds = [10**(-x)*.1 for x in range(N)]
    thresholds.append(0)
    if verbose:
        print(f"thresholds: {thresholds}")
    return thresholds

def extract_pruned_features(feature_importances, min_score=0.0):
    column_slice = feature_importances[feature_importances['weights'] > min_score]
    return column_slice.index.values

def features_selected_to_csv(thresholds, features_left):
    path = f"./selected_features_{time_produced}.csv"
    print(f"path: {path}")
    out_dat = pd.DataFrame(features_left , index=thresholds)
    out_dat.to_csv(path, index=False, header=True)

def plot_elbow(thresholds, num_features_left, auto_pick_elbow=True):
    if auto_pick_elbow:
        try: 
            from yellowbrick.utils import KneeLocator
            elbow_locator = KneeLocator(x=thresholds, y=num_features_left, curve_nature="convex", curve_direction="decreasing")
            best_threshold = elbow_locator.knee
            best_index_at = list(thresholds).index(best_threshold)
            best_num_feat = num_features_left[best_index_at]
        except:
            pass
    from matplotlib import rcParams
    plt.figure(figsize=(16, 6))
    #plt.xscale('log', nonposy='clip')
    rcParams.update({'font.size': 16})
    plt.plot(thresholds, num_features_left, '-o')
    if auto_pick_elbow:
        try:
            elbow_label = f"elbow at the={best_threshold} index_at: [{best_index_at}], best_num_feat={best_num_feat}"
            print(elbow_label)
            plt.vlines(best_threshold, color='r', linestyle="--", label=elbow_label, ymin = 0, ymax=len(non_zero_features))
            plt.legend(loc="best")
        except:
            pass
    plt.title('Feature Importance')
    plt.ylabel('Num of Features')
    plt.xlabel('Scores threshold')
    pic_name = f'num_feat_vs_score_threshold_{time_produced}.jpg'
    plt.savefig(pic_name)
    print(f"{pic_name} was produced.")
    
# @click.command(help='This command run feature selection on the input data saved in .csv')
# @click.argument('source')
# @click.argument('target')
# @click.argument('produce_importance_graph', required=False, default=True)
# @click.argument('produce_elbow_graph', required=False, default=True)

def feature_selection(source, target, produce_importance_graph=True):

    pd_data = pd.read_csv(source, encoding = 'big5')
    labeled_train = get_labeled_dat(pd_data, label=target)
    y_train = labeled_train.pop(target)
    X_train = labeled_train
    model = GradientBoostingRegressor()
    model.fit(X_train,y_train)

    features = list(X_train.columns)
    feature_importances = model.feature_importances_
    if produce_importance_graph:
        plot_feature_importances(features, feature_importances)
    feature_importances = feature_importances_to_csv(features, feature_importances)
    thresholds = get_thresholds()
    features_left = [extract_pruned_features(feature_importances, min_score=x) for x in thresholds]
    num_features_left = [len(x) for x in features_left]
    features_selected_to_csv(thresholds, features_left)
    plot_elbow(thresholds, num_features_left, auto_pick_elbow=True)

if __name__ == '__main__':
    source = "labeled_train_data_imputed_KNNImputer(add_indicator-False copy-True metric-nan_euclidean missing_values-nan n_neighbors-5 weights-distance).csv"
    target='R1COOH'
    feature_selection(source, target)