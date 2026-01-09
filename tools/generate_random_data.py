# generate_fake_data 不同的是，这种生成负样本的方法是随机选择未被填充的特征进行打乱，且每个样本打乱的特征不同
# generate_fake_data 没有考虑缺失值是否是填充状态，以及打乱的样本特征都相同

import pandas as pd
import os
import sys
import random
import numpy as np
from copy import copy
sys.path.append("run_OD")
sys.path.append("./")
from DetectOutlier.utils.file_io import PathManager
from DetectOutlier.utils.parameters import get_fake_dataname


random_seed = 42
repeat_times = 10


def random_shuffle_all_features(samples):
    feature_index = samples.columns.tolist()

    random.shuffle(feature_index)
    fake_samples=samples[feature_index]

    return fake_samples

def random_shuffle_features(samples, topk):
    feature_name = samples.columns.tolist()
    raw_feature_name = copy(feature_name)
    no_nan_name = samples.dropna(axis=1, how='all').columns.tolist()

    raw_topk_no_nan_name = no_nan_name[topk[0]: topk[1]]
    shuffle_topk_no_nan_name = copy(raw_topk_no_nan_name)
    random.shuffle(shuffle_topk_no_nan_name)
    # 获取原位置
    raw_index = [feature_name.index(_)  for _ in raw_topk_no_nan_name]
    for i in range(len(raw_index)):
        # print(f"pos: {raw_index[i]}:")
        # print(f"old peptide: {feature_name[raw_index[i]]}")
        feature_name[raw_index[i]] = shuffle_topk_no_nan_name[i]
        # print(f"new peptide: {feature_name[raw_index[i]]}")
    fake_samples = samples[feature_name]
    fake_samples.columns = raw_feature_name
    return fake_samples, shuffle_topk_no_nan_name

def generate_fake_data(data_name, fake_data, shuffle_ratio, repeat_time):
    print(f"process {data_name}-{shuffle_ratio}-{repeat_time}")
    feat_num = fake_data.shape[1]

    # 从0到int(feat_num * (1-shuffle_ratio))中随机调一个数
    C = random.randint(0, int(feat_num * (1-shuffle_ratio)))
    # 打乱C到 C+int(feat_num * shuffle_ratio)-1 位置的特征
    random_feat_index = (C, C+int(feat_num * shuffle_ratio)-1)

    fake_values, shuffle_topk_no_nan_name = random_shuffle_features(fake_data, random_feat_index)

    print(f"process {data_name}-{repeat_time} done")
    return fake_values, shuffle_topk_no_nan_name, f"fake-shuffle_ratio-{str(shuffle_ratio)}-repeat_time-{str(repeat_time)}"


def swap_and_track(df, shuffle_ratio=0.1):
    # 创建结果副本
    swapped_df = df.copy()

    # 存储交换信息：每行交换的列索引列表
    swap_info = []

    # 遍历每一行
    for i in range(len(df)):
        row = df.iloc[i]

        # 获取非NaN值的列索引
        non_nan_mask = row.notna()
        non_nan_cols = row.index[non_nan_mask].tolist()

        # 计算需要交换的元素数量（至少2个）
        n_swap = max(2, round(feat_shuffle_num))

        # 随机选择交换位置
        swap_cols = np.random.choice(non_nan_cols, size=int(n_swap), replace=False)

        # 获取交换值
        swap_values = swapped_df.iloc[i][swap_cols].values

        # 随机打乱这些值
        np.random.shuffle(swap_values)

        # 更新交换后的值
        swapped_df.iloc[i][swap_cols] = swap_values

        # 记录交换的列索引
        swap_info.append(list(swap_cols))

    return swapped_df, swap_info

if __name__ == '__main__':
    random.seed(random_seed)
    dataset_list_raw = ['HelaGroups.csv']
    data_name = dataset_list_raw[0].split(".")[0]

    format_data_dir = "../datasets/Clean"
    fake_data_dir = "../datasets/UltraFake"

    if not PathManager.exists(fake_data_dir):
        PathManager.mkdirs(fake_data_dir)

    clean_data = pd.read_csv(
        os.path.join(format_data_dir, f'{data_name}_clean_result.csv'),
        index_col=0
    )
    clean_data.dropna(how='all', axis=1, inplace=True)

    # shuffle for feature，特征打乱比例从 0.1 到 1
    # feature_shuffle_ratio = [_/100.0 for _ in range(1 , 41)]
    feature_shuffle_ratio = [_/1000.0 for _ in range(1, 21)]
    # 每一个比例打乱 repeat_times 次
    repeat_times = 10
    # 每次选择 clean_data / fake_groups 个样本
    fake_groups = 10

    for shuffle_ratio in feature_shuffle_ratio:
        for repeat_time in range(repeat_times):
            if repeat_time % fake_groups == 0:  # start from 0
                # reset random_seed
                random.seed(repeat_time)

                feat_shuffle_num = clean_data.shape[1] * shuffle_ratio
                non_nan_counts = clean_data.count(axis=1)
                for_fake_data = clean_data[non_nan_counts > feat_shuffle_num]
                fake_data = copy(for_fake_data)

                fake_num = fake_data.shape[0]
                group_fake_num = fake_num // fake_groups
                sample_index = [_ for _ in range(fake_num)]
                random.shuffle(sample_index)


            fake_files = os.path.join(
                fake_data_dir,
                f'{get_fake_dataname(shuffle_ratio, repeat_time)}'
            )

            start_ind = (repeat_time % fake_groups) * group_fake_num
            end_ind = (repeat_time % fake_groups + 1) * group_fake_num
            selected_values = fake_data.iloc[sample_index][start_ind: end_ind]
            # unselected_values1 = fake_data.iloc[sample_index][: start_ind]
            # unselected_values2 = fake_data.iloc[sample_index][end_ind: ]


            sample_fake_data, swap_info = swap_and_track(
                selected_values,
                feat_shuffle_num
            )
            sample_fake_data.index = "Fake_" + sample_fake_data.index
            # test_values = pd.concat([sample_fake_data, unselected_values1, unselected_values2], axis=0)
            test_values = pd.concat([sample_fake_data, clean_data], axis=0)
            sample_label = [1] * sample_fake_data.shape[0] + [0] * clean_data.shape[0]

            feature_name_list = test_values.columns.tolist()

            feature_outlier_label_list = []
            for isample_sawp_feat in swap_info:
                isample_feat_label = []
                for feature_name in feature_name_list:
                    if feature_name not in isample_sawp_feat:
                        isample_feat_label.append(0)
                    else:
                        isample_feat_label.append(1)
                feature_outlier_label_list.append(isample_feat_label)
            feature_label_list = np.concatenate((np.array(feature_outlier_label_list), np.zeros_like(clean_data)), axis=0)
            test_values.to_csv(f"{fake_files}.gz", compression='gzip')
            np.savez_compressed(f"{fake_files}.npz", feature_label=feature_label_list, sample_label=sample_label)
            print(f"{fake_files} done!!!")


