import openml as oml
import pandas as pd
import numpy as np
import os.path
from sklearn import preprocessing, ensemble

apikey = '5a0d8431b19613c12255e07c20568923'
oml.config.apikey = apikey
#Returns datasetsdict of dicts, or dataframe 返回OPenml上面的数据集列表
datasets = oml.datasets.list_datasets()
# 将数据集列表转化为Dataframe形式
datasets = pd.DataFrame(datasets)
# 将Dataframe形式转置
datasets = datasets.T
# 返回Dataframe里面数据集的元特征
#['MajorityClassSize', 'MaxNominalAttDistinctValues', 'MinorityClassSize','NumberOfClasses', 'NumberOfFeatures', 'NumberOfInstances',
#'NumberOfInstancesWithMissingValues', 'NumberOfMissingValues',
#'NumberOfNumericFeatures', 'NumberOfSymbolicFeatures', 'did', 'format',
#'name', 'status']
print(datasets.columns)

print(datasets[datasets['format'] == 'ARFF'].size)
"""
# 过滤一组数据集
filtered_dataset = datasets[datasets["NumberOfFeatures"] < 20]\
                           [datasets['NumberOfNumericFeatures']>0]\
                           [datasets['NumberOfClasses'] < 20]\
                           [datasets['NumberOfInstances'] > 50]

dids = list(filtered_dataset['did'].index)
"""

#final_ds = fd2[fd2['NumberOfInstances'] > 100]

def download_datasets(dids, path):
    i = 0

    for did in dids:
        try:

            if not os.path.isfile(str(did) + "-data"):
                ds = oml.datasets.get_dataset(did)
                #X,y ,z = ds.get_data(target=ds.default_target_attribute)
                X, y, categorical,attribute_names = ds.get_data(target=ds.default_target_attribute,
                                                                          include_row_id=True, dataset_format='array')
                np.save(path + "/" + str(did) + "-data", X)
                np.save(path + "/" + str(did) + "-target", y)
                np.save(path + "/" + str(did) + "-categorical", categorical)

        except:
            i = i+1
            print("Dataset with id " + str(did))

    print(str(i) + " datasets were impossible to download")

#保存为np的独有形式
#np.save("indexes", dids)


"""
"new_data = datasets[datasets['NumberOfClasses'] == 2]\
                   [datasets["NumberOfNumericFeatures"]>0]\
                   [datasets["MinorityClassSize"]>100]\
                   [datasets["NumberOfNumericFeatures"]<20]\
                   [datasets["NumberOfMissingValues"] < 20]
"""

#new_dids = list(new_data['did'].index)

#download_datasets(new_dids, "binary_problems")

#ds = oml.datasets.get_dataset(195)


# X - An array/dataframe where each row represents one example with
# the corresponding feature values.
# y - the classes for each example
# categorical_indicator - an array that indicates which feature is categorical
# attribute_names - the names of the features for the examples (X) and
# target feature (y)
#X, y, categorical_indicator, attribute_names = dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)


#X,Y,Z=ds.get_data(target=ds.default_target_attribute, return_categorical_indicator=True)

#from collections import Counter
#Counter(Y)

#np.where(np.array([1,2,3,4,4])==1)[0][0]???
#new_did_2 = datasets[datasets['NumberOfClasses'] == 2]\
#        [datasets["NumberOfNumericFeatures"]>-1]\
#        [datasets["MinorityClassSize"]>100]\
#        [datasets["NumberOfMissingValues"] == 0]["did"].values

filtered_dataset = datasets[datasets['NumberOfClasses'] == 2]\
        [datasets["NumberOfNumericFeatures"]>0]\
        [datasets["MinorityClassSize"]>100]\
        [datasets["NumberOfMissingValues"] == 0]

dids_3 = list(filtered_dataset['did'].index)

download_datasets(dids_3, "datasets/binary_numeric")
np.save("indexes_2",dids_3)



