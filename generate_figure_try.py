# set logger
import logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn", lineno=122)
#import h5py
import math

#from keras.models import Sequential, model_from_json
#from keras.layers import Dense, Dropout, Activation, regularizers, Flatten
#from keras.callbacks import CSVLogger
#cuML加速RF模型?
from sklearn import ensemble, preprocessing, multiclass
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, train_test_split
import seaborn as sns
from matplotlib import pyplot as plt

from collections import Counter


def load_dataset(id):
    X = np.load("datasets/binary_numeric/" + str(id) + "-data.npy")
    y = np.load("datasets/binary_numeric/" + str(id) + "-target.npy")
    categorical = np.load("datasets/binary_numeric/" + str(id) + "-categorical.npy")
    return X, y, categorical


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("---  new folder: " + path + "---")
        print("---  OK  ---")

    else:
        print("---  There is this folder!  ---")

#seed = 67
#np.random.seed(seed)
#too_big = 2500

#R = np.load("datasets/compressed/720-750trans7sam2500CV10/freq-720-750trans7sam2500CV10.npy")
#data_feature = pd.DataFrame(R)
#data_feature.columns = ["A", "B", "C"]


#   预处理生成图之前的数据集，清理掉非numerical的数据
def generate_picture(id, feature, too_big, shading, feature_colour, feature_linewidth, linewid, colour, dpinum, if_not_numerical, file):
    plt.clf()
    X, y, categorical = load_dataset(id)
    seed = 67
    np.random.seed(seed)
    if (X.shape[0] > too_big):
        new_indexes = np.random.choice(X.shape[0], too_big, replace=False)
        X = X[new_indexes]
        y = y[new_indexes]
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    X = imp.fit_transform(X)
    numerical_indexes = np.where(np.invert(categorical))[0]
    not_numerical_indexes = np.where(categorical)[0]
    print(numerical_indexes)
    print(not_numerical_indexes)
    X = X.T
    X = X.astype(float)
    y = y.T
    y = y.astype(float)
    #X_f = X[feature]
    #print(X[2])
    a = preprocessing.scale(X[feature])
    plt.xlim(-5, 5)
    plt.ylim(0, 1)
    sns.kdeplot(a, shade=shading, color=feature_colour, linewidth=feature_linewidth)
    c = preprocessing.scale(y)
    plot1 = sns.kdeplot(c, color='b', linewidth=1)
    not_num = not_numerical_indexes.tolist()
    not_num.append(feature)
    if if_not_numerical == 'N':
        X = np.delete(X, not_num, axis=0)
    elif if_not_numerical == 'Y':
        X = np.delete(X, [feature], axis=0)
    if not X.shape[0]:
        print("no more X : error")
        return
    if X.shape[0] > 60:
        linewid = 15 / X.shape[0]
    for i in range(X.shape[0]):
        b = preprocessing.scale(X[i])
        plot1 = sns.kdeplot(b, color=colour, linewidth=linewid)
    logger.info('生成图' + str(id) + '-' + str(feature))
    plot1out = plot1.get_figure()
    plot1out.savefig(file + '\\' + str(id) + '-' + str(feature)+'-dpi'+str(dpinum), dpi=dpinum)

"""
#global parameter
id_now = 31
X0, y0, categorical0 = generate_dataset(id_now)


def generate_picture(id, feature):
    global id_now, X0, y0, categorical0
    if id != id_now:
        X0, y0, categorical0 = generate_dataset(id)
        id_now = id
    for i in range(X0.shape[0]):
        if i == feature:
            a = preprocessing.scale(X0[i])
            sns.kdeplot(a, shade=True, color='r')
            continue
        a = preprocessing.scale(X0[i])
        plot1 = sns.kdeplot(a, color='k')
    logger.info('生成图' + str(id) + '-' + str(feature))
    plot1out = plot1.get_figure()
    plot1out.savefig('figure/' + str(id) + '-' + str(feature), dpi=400)
    
"""


def get_picture(data_feature, too_big, shading, feature_colour, feature_linewidth, linewid, colour, dpinum, if_not_numerical, file):
    for i in range(len(data_feature.A)):
        generate_picture(data_feature.A[i], data_feature.B[i], too_big, shading, feature_colour, feature_linewidth, linewid, colour, dpinum, if_not_numerical, file)


#get_picture()





# CREATING THE NEURAL NETS


