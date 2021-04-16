# CNN-based-AutoFE
code for CNN-based-AutoFE

## 1. 概述

该工程主要是分为几个文件依次实现。
他们分别是：
* main.py
  * 总体控制生成训练集、生成训练图片、设置CNN参数并调试
  * List of total useful binary data sets' IDs：下面包含的是目前已有的所有数据集的ID
  * settings for get target下面：dids代表的是准备生成一个新训练集的数据集的id，应从List of total useful binary data sets' IDs里选取
  * settings for get target下面：seed代表随机种子，尽量别动。too_big代表生成训练集允许的最大样本采样数量(太大会很慢)
  * settings for get target下面：transformations代表你要生成哪些变换的训练集，如果改动了，transformations也要相应改动
  * settings for generate figure下面：关于如何设置线宽  颜色  图片精度， 代码中有注释详细介绍
  * settings for CNN Trainer下面：设置CNN的参数，文件名，文件夹名设置，代码有详细注释
  * 运行栏输入相应函数即可运行相应模块
* CNN_based.py
  * 搭建CNN网络结构的地方，修改网络结构目前只能在这里修改
  * 包含了划分训练集，进行训练和验证等内容
  * 学习率，batch size等网络训练配置在main函数里设置
* get_target4.py
  * 生成新训练集的文件
* generate_feature_try.py
  * 生成图的文件
* data.py
  * 最开始从openml官网下载数据的文件
  
## 2.环境设置

具体的函数包如下：

下载openml数据集：
openml

python配置：

python 3.7.3

pandas 1.1.5

numpy  1.19.4

matplotlib  3.0.3

sklearn   0.24.0

torch   1.7.0+cu101

seaborn   0.11.1

pillow   5.4.1

'''
将来可能用到的数据增强包：
albumentations   0.5.2
```
