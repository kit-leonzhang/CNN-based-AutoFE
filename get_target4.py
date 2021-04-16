# set logger
import logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import pandas as pd
import numpy as np
#import h5py
import math

#from keras.models import Sequential, model_from_json
#from keras.layers import Dense, Dropout, Activation, regularizers, Flatten
#from keras.callbacks import CSVLogger
#cuML加速RF模型?
from sklearn import ensemble, preprocessing, multiclass
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, train_test_split

from collections import Counter


# Transformation
# 返回长度为len(col)的对col每个元素变换的列表
def sqrt(col):
    return list(map(np.sqrt, col));


def freq(col):
    col = np.floor(col)
    counter = Counter(col)
    return [counter.get(elem) for elem in col]


def tanh(col):
    return list(map(np.tanh, col));


def log(col):
    col1 = [i + 1e-6 for i in col]
    return list(map(np.log, col1));


def my_sigmoid(x):
    return 1 / (1 + math.exp(-x))


def sigmoid(col):
    return list(map(my_sigmoid, col))


def square(col):
    return list(map(np.square, col))


def normalize_val(num, col_min, col_max, norm_range):
    width = col_max - col_min

    if (width == 0): width = 1

    return (num - col_min) / width * (norm_range[1] - norm_range[0]) + norm_range[0]


def normalize(col):
    norm_range = (-1, 1)
    col_max = np.amax(col)
    col_min = np.amin(col)

    return list(map(lambda x: normalize_val(x, col_min, col_max, norm_range), col))

# Globals

# Datasets
#dids = np.load("datasets/indexes.npy")
#dids = np.load("indexes_2.npy")

#dids = [31, 37]

#dids = [ 913,   914,   915,   917,   918,   920,   923,   925,   926,
#         931,   933,   934,   935,   936,   937,   943,   958,   962,
#         970,   971,   976,   977,   978,   979,   980,   983,   994]
"""
dids = [ 246,   251,   256,   257,   258,   260,   262,   264,
         267,   269,   273,   293,   310,   312,   350,   351,   354,
         357,   464,   715,   717,   718,   720,   722,   723,   724,
         725,   727,   728,   730,   732,   734,   735,   737,   740,
         741,   742,   743,   744,   746,   749,   750,   751,   752,
         761,   763,   766,   769,   770,   772,   773,   774,   776,
         778,   779,   792,   793,   794,   795,   797,   799,   803,
         805,   806,   807,   811,   813,   814,   816,   818,   819,
         821,   823,   824,   825,   827,   832,   833,   834,   837,
         838,   841,   843,   845,   846,   847,   849,   851,   853,
         855,   860,   863,   866,   869,   870,   871,   873,   877,
         879,   880,   881,   884,   886,   888,   896,   900,   901,
         903,   904,   906,   907,   908,   909,   910,   911,   912,
         913,   914,   915,   917,   918,   920,   923,   925,   926,
         931,   933,   934,   935,   936,   937,   943,   958,   962,
         970,   971,   976,   977,   978,   979,   980,   983,   994,
         995,   997,  1011,  1019,  1020,  1021,  1038,  1039,  1040,
        1042,  1046,  1048,  1049,  1050,  1063,  1067,  1073,  1116,
        1120,  1126,  1128,  1129,  1130,  1134,  1136,  1137,  1138,
        1140,  1145,  1148,  1149,  1150,  1153,  1158,  1160,  1161,
        1162,  1163,  1165,  1166,  1167,  1169,  1181,  1182,  1205,
        1212,  1216,  1217,  1218,  1219,  1220,  1235,  1241,  1242,
        1444,  1453,  1460,  1461,  1462,  1464,  1471,  1479,  1480,
        1485,  1486,  1487,  1489,  1494,  1496,  1498,  1502,  1504,
        1507,  1510,  1511,  1547,  1558,  1566,  1597,  4134,  4136,
        4137, 23517, 40514, 40589, 40592, 40594, 40595, 40597, 40665,
       40666, 40701, 40704, 40705, 40710, 40922, 40978, 40981, 40983,
       40999, 41005, 41007, 41026, 41142, 41143, 41144, 41145, 41146]
"""

"""
# RF model parameters
seed = 67
np.random.seed(seed)
#too_big = 10000
too_big = 2500
#transformations = [sqrt, freq]
#transformations_name = ["sqrt", "freq"]
transformations = [sqrt, freq, tanh, log, sigmoid, square, normalize]
transformations_name = ["sqrt", "freq","tanh", "log", "sigmoid", "square", "normalize"]
trans2target1 = {}
trans2target2 = {}
trans2target3 = {}
"""


# def binarize_dataset():

def load_dataset(id):
    X = np.load("datasets/binary_numeric/" + str(id) + "-data.npy")
    y = np.load("datasets/binary_numeric/" + str(id) + "-target.npy")
    categorical = np.load("datasets/binary_numeric/" + str(id) + "-categorical.npy")
    return X, y, categorical


def evaluate_model(X, y, categorical, seed):
    # 填补缺失值 imputer
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    X = imp.fit_transform(X)
    enc = preprocessing.OneHotEncoder()
    X = enc.fit_transform(X)
    clf = ensemble.RandomForestClassifier(random_state=seed)
    # clf_ovsr = multiclass.OneVsRestClassifier(clf, n_jobs=-1)

    #return cross_val_score(clf, X, y, cv=10)
    logger.info('cross_val_score(clf, X, y, cv=10)')
    return cross_val_score(clf, X, y, cv=10)



def is_positive(X, y, categorical, base_score, transformation, feature , seed):
    transformed_feature = np.array(transformation(X[:, feature]))
    # 把老X和变换的并排放？
    X = np.c_[X, transformed_feature]
    categorical = np.append(categorical, False)
    new_score = evaluate_model(X, y, categorical, seed).mean()

    #return 1 if (base_score <= (new_score - 0.01)) else 0
    return 1 if (new_score > base_score * 1.01) else 0

def is_positive_2(X, y, categorical, base_score, transformation, feature):
    transformed_feature = np.array(transformation(X[:, feature]))
    new_score = evaluate_model(transformed_feature.reshape(-1, 1), y, [False]).mean()

    return 1 if (base_score <= (new_score - 0.005)) else 0


def is_positive_3(X, y, categorical, base_score, transformation, feature):
    transformed_feature = np.array(transformation(X[:, feature]))
    new_score = evaluate_model(transformed_feature.reshape(-1, 1), y, [False]).mean()

    return 1 if (new_score > base_score * 1.01) else 0


# Build the target for the compressed feature
#bad_datasets = []


def build_target_for_compressed(dids, trans2target1, trans2target2, trans2target3, transformations, too_big,
                                bad_datasets, seed):
    for transf in transformations:
        trans2target1[transf] = []
        trans2target2[transf] = []
        trans2target3[transf] = []

    for did in dids:
        print("Start dataset number", did)
        print(bad_datasets)

        try:

            X, y, categorical = load_dataset(did)

            new_indexes = []

            if (X.shape[0] > too_big):
                new_indexes = np.random.choice(X.shape[0], too_big, replace=False)
                X = X[new_indexes]
                y = y[new_indexes]
            logger.info('base_score = evaluate_model(X, y, categorical).mean()')
            base_score = evaluate_model(X, y, categorical, seed).mean()

            # Find the indexes of numeric attributes
            numerical_indexes = np.where(np.invert(categorical))[0]
            # 为什么这里要随机选取特征值？
            sample_numerical_indexes = np.random.choice(numerical_indexes, min(numerical_indexes.shape[0], 10),
                                                        replace=False)
            # sample_numerical_indexes = np.random.choice(numerical_indexes, numerical_indexes.shape[0],
            #                                                replace = False)

            for i, transf in enumerate(transformations):
                for feature in sample_numerical_indexes:
                    print("\tEvaluating feature " + str(feature))
                    logger.info('mlp_target_1 = is_positive(X, y, categorical, base_score, transf, feature)')
                    mlp_target_1 = is_positive(X, y, categorical, base_score, transf, feature, seed)
                    # mlp_target_2 = is_positive_2(X, y, categorical, base_score, transf, feature)
                    # mlp_target_3 = is_positive_3(X, y, categorical, base_score, transf, feature)

                    print("\t\t" + str(mlp_target_1))
                    # print("\t\t" + str(mlp_target_1), str(mlp_target_2), str(mlp_target_3))

                    trans2target1[transf].append((did, feature, mlp_target_1))
                    # trans2target2[transf].append((did, feature, mlp_target_2))
                    # trans2target3[transf].append((did, feature, mlp_target_3))

        except:
            print("The evaluation of dataset " + str(did) + " failed")
            bad_datasets.append(did)
            continue


def save_target_for_compressed(path, transformations, transformations_name, trans2target1):

    for transf, name in zip(transformations, transformations_name):
        np.save(path + name + "-913-994trans7sam2500CV10", trans2target1[transf])
        #np.save(path + name + "-2", trans2target2[transf])
        #np.save(path + name + "-3", trans2target3[transf])


"""
logger.info('搭建压缩版数据集的target')
build_target_for_compressed(dids)
logger.info('保存压缩版数据集的target')
save_target_for_compressed("datasets/compressed/913-994trans7sam2500CV10/")
"""

#build_target_for_compressed(dids)
