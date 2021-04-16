import logging
import get_target4
import generate_figure_try
import CNN_based
import numpy as np
import pandas as pd
import os
from warnings import simplefilter
# set logger
simplefilter(action='ignore', category=FutureWarning)
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Todo ======= List of total useful binary data sets' IDs =======


useful_binary_datasets_id = [31, 37, 53, 59, 151, 152, 153, 161, 162, 246, 251, 256, 257, 258, 260, 262, 264, 267, 269, 310, 312, 464, 715, 717, 718, 720, 722, 723, 724,
725, 727, 728, 730, 732, 734, 735, 737, 740, 741, 742, 743, 744, 746, 749, 750, 751, 752, 761, 763, 766, 769, 770, 772, 773, 774, 776, 778,
779, 792, 793, 794, 795, 797, 799, 803, 805, 806, 807, 811, 813, 814, 816, 818, 819, 821, 823, 824, 825, 827, 832, 833, 834, 837, 838, 841,
843, 845, 846, 847, 849, 851, 853, 855, 860, 863, 866, 869, 870, 871, 873, 877, 879, 880, 881, 884, 886, 888, 896, 900, 901, 903, 904, 906,
907, 908, 909, 910, 911, 912, 913, 914, 915, 917, 918, 920, 923, 925, 926, 931, 933, 934, 935, 936, 937, 943, 958, 962, 970, 971, 976, 977,
978, 979, 980, 983, 994, 995, 997, 1011, 1019, 1021, 1046, 1048, 1049, 1050, 1063, 1067, 1073, 1120, 1167, 1169, 1182, 1205, 1216, 1217,
1218, 1219, 1220, 1235, 1444, 1453, 1460, 1461, 1462, 1464, 1471, 1480, 1489, 1494, 1496, 1498, 1502, 1504, 1507, 1510, 1511, 1547, 1558,
1597, 23517, 40514, 40701, 40704, 40705, 40710, 40922, 40981, 40983, 40999, 41007, 41146, 41434, 41672, 41945, 41946, 42192, 42193,
42344, 42397, 42477, 42493, 42680, 42717, 42732, 42733, 42769]


# TODO ======= settings for get target =======

# dids列表 是 即将生成新训练集 的 数据集id列表，填入相应的id，设置好参数，在main里运行get_target()即可生成相应训练集
dids = [913,   914,   915,   917,   918,   920,   923,   925,   926,
         931,   933,   934,   935,   936,   937,   943,   958,   962,
         970,   971,   976,   977,   978,   979,   980,   983,   994]

# RF model parameters
seed = 67          # 随机种子，尽量别动，暂时所有训练集都是这个随机种子训练出来的
np.random.seed(seed)

too_big = 2500     # random forest 打分用的数据集训练数的上限

transformations = [get_target4.sqrt, get_target4.freq, get_target4.tanh, get_target4.log, get_target4.sigmoid,
                   get_target4.square, get_target4.normalize]       #选择变换的列表，目前只有7个一维变换

transformations_name = ["sqrt", "freq", "tanh", "log", "sigmoid", "square", "normalize"]        #变换名字的列表

# TODO ======= settings for generate figure =======
R = np.load("datasets/compressed/751-803trans7sam2500CV10/freq-751-803trans7sam2500CV10.npy")   # 选择你要生成图的训练集
shading = True      # feature 线处是否有阴影
feature_colour = 'r'
feature_linewidth = 2
colour = 'k'
linewid = 0.25
dpinum = 6
# dpi == 6 对应的是 38*28
# dpi == 200 对应的是1280*960
if_not_numerical = 'N'
path = os.getcwd()
file = path + "\\figure1"

# TODO ======= settings for CNN Trainer =======
seed = 67
lr_set = 5e-5
batch_size_set = 8
num_epochs_set = 500
target_path_set = "target/"
path_set = "figure1/"
save_name_best_set = 'model_state/'+'batch_size'+str(batch_size_set) +'ckpt_best.pt'
save_name_end_set = 'model_state/ckpt_routine_'+str(num_epochs_set)+'.pt'



# TODO ======= below is irrelevant  以下部分不需要设置，直接在main后面选择相应的函数运行即可 =======
trans2target1 = {}
trans2target2 = {}
trans2target3 = {}
bad_datasets = []
data_feature = pd.DataFrame(R)
data_feature.columns = ["A", "B", "C"]
tryfeature = data_feature.B



def get_target():
    logger.info('搭建压缩版数据集的target')
    get_target4.build_target_for_compressed(dids, trans2target1, trans2target2, trans2target3, transformations,
                                            too_big, bad_datasets, seed)
    logger.info('保存压缩版数据集的target')
    get_target4.save_target_for_compressed("datasets/compressed/913-994trans7sam2500CV10/",
                                           transformations, transformations_name, trans2target1)


if __name__ == "__main__":
    #generate_figure_try.mkdir(file)
    #generate_figure_try.get_picture(data_feature,too_big, shading, feature_colour, feature_linewidth, linewid, colour, dpinum, if_not_numerical, file)
    #print(min(tryfeature))
    CNN_based.train(seed, lr_set, batch_size_set, num_epochs_set, path_set, target_path_set, save_name_best_set, save_name_end_set)