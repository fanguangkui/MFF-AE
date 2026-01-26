import numpy as np
import pandas as pd
import random
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
import platform
from copy import copy
from DetectOutlier.utils.data import get_distance, get_scale, get_global

class LabelDataGenerator(object):
    def __init__(self, args, infer_flag=False):

        if platform.system().lower() == 'windows':
            self.R_USER = args.Windows_R_USER
            self.R_HOME = None
        else:
            self.R_USER = args.Linux_R_USER
            self.R_HOME = args.Linux_R_HOME

        # labeled_outlier = []
        # for i in args.external_outlier_candidates:
        #     labeled_outlier += list(i.values())[0]
        # self.labeled_outlier = list(set(labeled_outlier))

        self.sample_ratio = args.sample_ratio
        self.generate_duplicates = args.generate_duplicates
        self.input_matrix = args.input_matrix

        self.joint_data = args.joint_data
        self.fill_strategy = args.fill_strategy
        self.scale_strategy = args.scale_strategy

        if not infer_flag:
            self.prepare_data_dir = args.prepare_data_dir
            self.dataset_list = args.dataset_list_raw

    def preprocessor(self, data_dir=None, dataset_file=None, raw_data=None, dropna=True):
        # in deep learning, should drop the nan of column
        if (raw_data is None) and (data_dir is not None) and (dataset_file is not None):
            raw_data = pd.read_csv(os.path.join(data_dir, dataset_file), index_col=0)
        assert raw_data is not None, f"no data to process, check {data_dir} and {dataset_file}"

        if dropna:
            column_nums = raw_data.shape[1]
            raw_data.dropna(how='all', axis=1, inplace=True)
            print(f"before dropnan columns: {column_nums}, now columns count is {raw_data.shape[1]}")

        scale_data = get_scale(raw_data, self.scale_strategy, r_user=self.R_USER, r_home=self.R_HOME)
        fill_mask = (~raw_data.isna()).astype(int).values
        index_list = raw_data.index.tolist()
        column_list = raw_data.columns.tolist()

        scale_data = pd.DataFrame(
            data=scale_data,
            index=index_list,
            columns=column_list
        )
        scale_data = get_global(scale_data, strategy=self.fill_strategy)
        return scale_data, fill_mask

    def generator(self, dataset, kf_num=0, saved_dir=None, TestonAll=True, **kwargs):
        '''
        la: labeled anomalies, can be either the ratio of labeled anomalies or the number of labeled anomalies
        at_least_one_labeled: whether to guarantee at least one labeled anomalies in the training set
        https://stats.stackexchange.com/questions/387326/unsupervised-learning-train-test-division
        '''

        # load dataset
        feature_data = dataset
        dataname=kwargs.get("dataset_file", "infer_data.csv").split(".gz")[0]
        # spliting the current data to the training set and testing set
        index_list = feature_data.index.tolist()
        column_list = feature_data.columns.tolist()
        in_data = feature_data.values
        print(in_data.shape)
        # for ioutlier in self.labeled_outlier:
        #     outlier_ind = index_list.index(ioutlier)
        #     data_label[outlier_ind] = 1
        data_label = kwargs.get('data_label', np.zeros(dataset.shape[0]))
        fill_mask = kwargs.get('fill_mask', np.zeros_like(dataset))
        sample_index_id = np.arange(0, len(index_list))

        if TestonAll:
            data_dict = {
                'X_train': in_data,
                'y_train_index': np.array(index_list),
                'y_train_label': data_label,
                'X_test': in_data,
                'y_test_index': np.array(index_list),
                'y_test_label': data_label,
                "sample_index_id": sample_index_id,
                "column_list": column_list
            }
        else:
            # 获取标签0和标签1的索引
            normal_indices  = np.where(data_label == 0)[0]
            anomaly_indices  = np.where(data_label == 1)[0]
            # 将异常样本平均分配到训练集和测试集
            np.random.seed(0)
            np.random.shuffle(anomaly_indices)
            _anomalies = int(len(anomaly_indices) * 0.3)
            anomaly_test = anomaly_indices[:_anomalies]
            anomaly_train = anomaly_indices[_anomalies:]

            # 将正常样本分配到训练集和测试集
            normal_train, normal_test = train_test_split(
                normal_indices,
                test_size=int(len(normal_indices) * 0.3),
                random_state=0
            )

            # 组合索引
            train_indices = np.concatenate([normal_train, anomaly_train])
            test_indices = np.concatenate([normal_test, anomaly_test])

            # 打乱顺序
            np.random.shuffle(train_indices)
            np.random.shuffle(test_indices)


            data_dict = {
                'X_train':in_data[train_indices],
                'y_train_index':sample_index_id[train_indices],
                'y_train_label':data_label[train_indices],
                'X_test':in_data[test_indices],
                'y_test_index':sample_index_id[test_indices],
                'y_test_label':data_label[test_indices],
                "sample_index_id": sample_index_id[test_indices]
            }
        if saved_dir is not None:
            prefix = f"seed_list-1-dataset_list-{dataname}"
            np.save(
                os.path.join(saved_dir, f'{prefix}.npy'),
                data_dict
            )
        return data_dict




