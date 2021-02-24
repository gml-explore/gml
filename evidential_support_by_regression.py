from copy import copy
from scipy.sparse import *
from scipy.stats import t
from sklearn.linear_model import LinearRegression
import logging
import numpy as np
import gml_utils
from pyds import MassFunction


class Regression:
    '''
    线性回归相关类，对所有feature进行线性回归
    todo: feature回归的更新策略:只回归证据支持有变化的feature
    '''
    def __init__(self, each_feature_easys, n_job,effective_training_count_threshold =2):
        '''
        初始化
        @param each_feature_easys:
        @param n_job:
        @param effective_training_count_threshold:
        '''
        self.effective_training_count = max(2, effective_training_count_threshold)
        self.n_job = n_job
        if len(each_feature_easys) > 0:
            XY = np.array(each_feature_easys)
            self.X = XY[:, 0].reshape(-1, 1)
            self.Y = XY[:, 1].reshape(-1, 1)
        else:
            self.X = np.array([]).reshape(-1, 1)
            self.Y = np.array([]).reshape(-1, 1)
        self.balance_weight_y0_count = 0
        self.balance_weight_y1_count = 0
        for y in self.Y:
            if y > 0:
                self.balance_weight_y1_count += 1
            else:
                self.balance_weight_y0_count += 1
        self.perform()

    def perform(self):
        '''
        执行线性回归
        @return:
        '''
        self.N = np.size(self.X)
        if self.N <= self.effective_training_count:
            self.regression = None
            self.residual = None
            self.meanX = None
            self.variance = None
            self.k = None
            self.b = None
        else:
            sample_weight_list = None
            if self.balance_weight_y1_count > 0 and self.balance_weight_y0_count > 0:
                sample_weight_list = list()
                sample_weight = float(self.balance_weight_y0_count) / self.balance_weight_y1_count
                for y in self.Y:
                    if y[0] > 0:
                        sample_weight_list.append(sample_weight)
                    else:
                        sample_weight_list.append(1)
            self.regression = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=self.n_job).fit(self.X, self.Y,
                                                                                                       sample_weight=sample_weight_list)
            self.residual = np.sum((self.regression.predict(self.X) - self.Y) ** 2) / (self.N - 2)
            self.meanX = np.mean(self.X)  # 此feature的所有证据变量的feature_value的平均值
            self.variance = np.sum((self.X - self.meanX) ** 2)
            z = self.regression.predict(np.array([0, 1]).reshape(-1, 1))
            self.k = (z[1] - z[0])[0]
            self.b = z[0][0]



def create_csr_matrix(variables,features):
    '''
    创建稀疏矩阵存储所有variable的所有featureValue，用于后续计算Evidential Support
    :return:
    '''
    data = list()
    row = list()
    col = list()
    # 统计函数化相关的变量和特征的个数
    var_len = 0
    fea_len = 0
    for fea in features:
        if fea['parameterize'] == 1:
            fea_len += 1
    for index, var in enumerate(variables):
        count = 0
        feature_set = variables[index]['feature_set']
        for feature_id in feature_set:
            if features[feature_id]['parameterize'] == 1:
                count += 1
        if count > 0:
            var_len += 1

    # print('var_len', var_len)
    # print('fea_len', fea_len)
    for index, var in enumerate(variables):
        feature_set = variables[index]['feature_set']
        for feature_id in feature_set:
            if features[feature_id]['parameterize'] == 1:
                data.append(feature_set[feature_id][1] + 1e-8)
                row.append(index)
                col.append(feature_id)
    return csr_matrix((data, (row, col)), shape=(var_len, fea_len))




