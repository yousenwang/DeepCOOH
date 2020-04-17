"""
Author: You Sen Wang (Ethan)
Started Date: 04/13/2020
Email: yousenwang@gmail.com
"""

import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
from scipy import stats
from scipy.stats import norm, skew

def plot_norm(feat_series, extend_name):
    fig = plt.figure()
    sns.distplot(feat_series , fit=norm);
    # Get the fitted parameters used by the function
    (mu, sigma) = norm.fit(feat_series)
    print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

    #Now plot the distribution
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
                loc='best')
    plt.ylabel('Frequency')
    plt.title(f'{feat_series.name} {extend_name} distribution')
    pic_name = f'{feat_series.name}_{extend_name}_distribution.jpg'
    plt.savefig(pic_name)
    print(f"{pic_name} was produced.")


def plot_QQ(feat_series, extend_name):
    #Get also the QQ-plot
    fig = plt.figure()
    res = stats.probplot(feat_series, plot=plt)
    plt.title(f'{feat_series.name} {extend_name} QQ plot')
    pic_name = f'{feat_series.name}_{extend_name}_QQ_plot.jpg'
    plt.savefig(pic_name)
    print(f"{pic_name} was produced.")

def data_correlation(train):
    corrmat = train.corr()
    plt.subplots(figsize=(12*3,9*3))
    sns.heatmap(corrmat, vmax=0.9, square=True)
    pic_name = f'data_correlation.jpg'
    plt.savefig(pic_name)
    print(f"{pic_name} was produced.")

def plot_tuning_graph(y_true, y_pred, final_model, features, grid_result=None):
    from sklearn.metrics import r2_score
    fin_r2 = r2_score(y_true, y_pred)

    import matplotlib.pyplot as plt
    import datetime
    plt.figure()
    n_y = len(y_pred)
    y_mean = [y_true.mean() for x in range(n_y)]
    xs = [x for x in range(n_y)]
    plt.plot(xs,y_true, label = "True")
    plt.plot(xs,y_pred, label = f"{final_model.__class__.__name__} {fin_r2}", linestyle='dashed')
    plt.plot(xs,y_mean, label = "mean", linestyle='dashed')
    plot_title = f"Num of Feats: {len(features)} "
    if grid_result!=None:
        plot_title+=f"Best: {grid_result.best_score_} \n using {grid_result.best_params_}"
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
