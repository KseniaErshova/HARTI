# Load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import datetime
import tqdm
import os
import glob
import pickle
import json
import platform
import matplotlib.mlab as mlab
from decimal import Decimal
from collections import OrderedDict
import warnings
from itertools import combinations
import seaborn as sns

import lifelines
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.plotting import add_at_risk_counts
from lifelines import CoxPHFitter, NelsonAalenFitter
from lifelines.utils import k_fold_cross_validation
from lifelines.utils import survival_table_from_events
from lifelines.utils import restricted_mean_survival_time
from lifelines.statistics import survival_difference_at_fixed_point_in_time_test
from lifelines.utils import covariates_from_event_matrix
from lifelines.utils import to_long_format
from lifelines.utils import add_covariate_to_timeline
from lifelines import CoxTimeVaryingFitter

import scipy.stats as stats
from scipy.stats import chi2_contingency as chi
from scipy.stats import t as ttest
from scipy.stats import ttest_ind as ttest_ind
from scipy.stats import linregress
from scipy.stats import kruskal as kruskal


import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats import multitest
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint as ci
from statsmodels.stats.weightstats import DescrStatsW
from statsmodels.tsa.stattools import kpss, adfuller
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA

import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans as km
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.cluster import DBSCAN, KMeans
from sklearn.manifold import TSNE
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score, precision_recall_fscore_support, precision_recall_curve, f1_score, classification_report
from sklearn.exceptions import DataConversionWarning
kmeans = KMeans(n_clusters=3)

from lightgbm import LGBMClassifier
try:
    from xgboost import XGBClassifier
    from xgboost import plot_importance
    from xgboost import plot_tree
except ModuleNotFoundError as e:
    print(f"Cannot import XGBOOST: {e}")

pd.set_option('display.max_columns', None)
sns.set(style="white")
#plt.rc('font',family='DejaVu Sans', size=12)
plt.rc('font',family='Times New Roman', size=20)



# --------------------- Functions
# MinMax scaler
def minmaxscale(x):
    return (x - min(x)) / (max(x) - min(x))

# Define info function
def print_info(d):
    print("Number of Rows:\t\t", d.shape[0])
    print("Number of Columns:\t", d.shape[1])
    print("Unique IDs:\t\t", d.ID.unique().shape[0])
    print("Day in ICU min:\t\t", d.day_in_icu.min())
    print("Columns:\t\t\n", d.columns.values)


def replace_from_dict(val, dict_):
    return dict_[val] if val in dict_ else val


def train_and_predict(clf, x_train, y_train, x_val, y_val, columns=None, keep=True, return_probs=False, return_roc_auc=True):
    if isinstance(columns, list) and not keep:
        columns = x_train.columns[~x_train.columns.str.contains("|".join(columns))]
    
    if columns is not None:
        clf.fit(x_train[columns], y_train.astype('int'))
        y_pred = clf.predict(x_val[columns])
        y_pred_proba = clf.predict_proba(x_val[columns])
    else:
        clf.fit(x_train, y_train.astype('int'))
        y_pred = clf.predict(x_val)
        y_pred_proba = clf.predict_proba(x_val)
    
    
    result = {}
    
    result['rocauc'] = roc_auc_score(y_val.astype('int'), y_pred_proba[:,1])
    result['accuracy'] = accuracy_score(y_val.astype('int'), y_pred)
    result['bce'] = log_loss(y_val.astype('int'), y_pred_proba)
    result['y_val'] = y_val
    result['y_pred'] = y_pred
    result['proba'] = y_pred_proba
    
    return result

def print_covariance_heatmap(X):
    """Visualize correlation between every two factors
    Plot beautiful Covariance matrix heatmap
    X: 'DataFrame'
    """
    corr = X.corr()
        # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(20, 20))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
