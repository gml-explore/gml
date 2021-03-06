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

class EvidentialSupport:
    '''
    Evidential Support计算类
    '''
    def __init__(self,variables,features,method = 'regression',evidence_interval_count=10):
        self.variables = variables
        self.features = features
        self.features_easys = dict()  # 存放所有features的所有easy的featurevalue   :feature_id:[[value1,bound],[value2,bound]...]
        self.tau_and_regression_bound = 10
        self.evidence_interval_count = evidence_interval_count  #区间数为10
        self.NOT_NONE_VALUE = 1e-8
        self.n_job = 10
        self.delta = 2
        self.effective_training_count_threshold = 2
        self.word_evi_uncer_degree = 0.4
        self.relation_evi_uncer_degree = 0.1
        self.observed_variables_set, self.poential_variables_set = gml_utils.separate_variables(self.variables)
        #根据evidential support方法的不同，分别做不同的初始化
        if method == 'regression':
            self.data_matrix = self.create_csr_matrix()
            self.evidence_interval = gml_utils.init_evidence_interval(self.evidence_interval_count)
            gml_utils.init_evidence(self.features,self.evidence_interval,self.observed_variables_set)
            gml_utils.init_bound(self.variables,self.features)
        if method =='relation':
            self.dict_rel_acc = self.get_dict_rel_acc()

    def get_unlabeled_var_feature_evi(self):
        '''
        计算每个隐变量的每个unary feature相关联的证据变量里面0和1的比例，以及binaryfeature另一端的变量id,
        最终向每个隐变量增加nary_feature_evi_prob和binary_feature_evi两个属性
        :return:
        '''
        self.observed_variables_set, self.poential_variables_set = gml_utils.separate_variables(self.variables)
        unary_feature_evi_prob = list()
        binary_feature_evi = list()
        for id in self.poential_variables_set:
                for (feature_id,value) in self.variables[id]['feature_set'].items():
                    feature_name = self.features[feature_id]['feature_name']
                    weight = self.features[feature_id]['weight']
                    if self.features[feature_id]['feature_type'] == 'unary_feature':
                        labeled0_vars_num = 0
                        labeled1_vars_num = 0
                        for var_id in weight:
                            if self.variables[var_id]['is_evidence'] == True and self.variables[var_id]['label'] == 0:
                                labeled0_vars_num += 1
                            elif self.variables[var_id]['is_evidence'] == True and self.variables[var_id]['label'] == 1:
                                labeled1_vars_num += 1
                        if labeled1_vars_num==0 and labeled0_vars_num == 0:
                            continue
                        n_samples = labeled0_vars_num + labeled1_vars_num
                        neg_prob = float(labeled0_vars_num/n_samples)
                        pos_prob = float(labeled1_vars_num/n_samples)
                        unary_feature_evi_prob.append((feature_id,feature_name,n_samples,neg_prob,pos_prob))
                    elif self.features[feature_id]['feature_type'] == 'binary_feature':
                        for (id1,id2) in weight:
                            anotherid = id1 if id1!=id else id2
                            if self.variables[anotherid]['is_evidence'] == True:
                                binary_feature_evi.append((anotherid,feature_name,feature_id))
                self.variables[id]['unary_feature_evi_prob'] = copy(unary_feature_evi_prob)
                self.variables[id]['binary_feature_evi'] = copy(binary_feature_evi)
                unary_feature_evi_prob.clear()
                binary_feature_evi.clear()

    def separate_feature_value(self):
        '''
        选出每个feature的easy feature value用于线性回归
        :return:
        '''
        each_feature_easys = list()
        self.features_easys.clear()
        for feature in self.features:
            each_feature_easys.clear()
            for var_id, value in feature['weight'].items():
                # 每个feature拥有的easy变量的feature_value
                if var_id in self.observed_variables_set:
                    each_feature_easys.append([value[1], (1 if self.variables[var_id]['label'] == 1 else -1) * self.tau_and_regression_bound])
            self.features_easys[feature['feature_id']] = copy(each_feature_easys)

    def create_csr_matrix(self):
        '''
        创建稀疏矩阵存储所有variable的所有featureValue，用于后续计算Evidential Support
        :return:
        '''
        data = list()
        row = list()
        col = list()
        for index, var in enumerate(self.variables):
            feature_set = self.variables[index]['feature_set']
            for feature_id in feature_set:
                data.append(feature_set[feature_id][1] + self.NOT_NONE_VALUE)
                row.append(index)
                col.append(feature_id)
        return csr_matrix((data, (row, col)), shape=(len(self.variables), len(self.features)))

    def influence_modeling(self,update_feature_set):
        '''
        对已更新feature进行线性回归
        @param update_feature_set:
        @return:
        '''
        if len(update_feature_set) > 0:
            self.init_tau_and_alpha(update_feature_set)
            #logging.info("init para finished")
            for feature_id in update_feature_set:
                # 对于某些features_easys为空的feature,回归后regression为none
                self.features[feature_id]['regression'] = Regression(self.features_easys[feature_id], n_job=self.n_job)
            #logging.info("feature regression finished")

    def init_tau_and_alpha(self, feature_set):
        '''
        对给定的feature计算tau和alpha
        @param feature_set:
        @return:
        '''
        if type(feature_set) != list and type(feature_set) != set:
            print("输入参数错误，应为set或者list")
            return
        else:
            for feature_id in feature_set:
                # tau值固定为上界
                self.features[feature_id]["tau"] = self.tau_and_regression_bound
                weight = self.features[feature_id]["weight"]
                labelvalue0 = 0
                num0 = 0
                labelvalue1 = 0
                num1 = 0
                for key in weight:
                    if self.variables[key]["is_evidence"] and self.variables[key]["label"] == 0:
                        labelvalue0 += weight[key][1]
                        num0 += 1
                    elif self.variables[key]["is_evidence"] and self.variables[key]["label"] == 1:
                        labelvalue1 += weight[key][1]
                        num1 += 1
                if num0 == 0 and num1 == 0:
                    continue
                if num0 == 0:
                    # 如果没有和该feature相连的label0，就把值赋值为value的上界，目前先定为1
                    labelvalue0 = 1
                else:
                    # label为0的featurevalue的平均值
                    labelvalue0 /= num0
                if num1 == 0:
                    # 同上
                    labelvalue1 = 1
                else:
                    # label为1的featurevalue的平均值
                    labelvalue1 /= num1
                alpha = (labelvalue0 + labelvalue1) / 2
                self.features[feature_id]["alpha"] = alpha

    def evidential_support_by_regression(self,variable_set,update_feature_set):
        '''
        计算给定隐变量集合的Evidential Support,适用于ER
        @param variable_set:
        @param update_feature_set:
        @return:
        '''
        self.observed_variables_set, self.poential_variables_set = gml_utils.separate_variables(self.variables)
        self.separate_feature_value()
        if update_feature_set == None or (type(update_feature_set) ==set() and len(update_feature_set) == 0):
            update_feature_set = set()
            for fid,feature in enumerate(self.features):
                update_feature_set.add(fid)
        self.influence_modeling(update_feature_set)
        coo_data = self.data_matrix.tocoo()
        row, col, data = coo_data.row, coo_data.col, coo_data.data
        coefs = []
        intercept = []
        residuals = []
        Ns = []
        meanX = []
        variance = []
        delta = self.delta
        zero_confidence = []
        for feature in self.features:
            if feature['regression'].regression is not None and feature['regression'].regression.coef_[0][0] >=0:
                feature['monotonicity'] = True
            else:
                feature['monotonicity'] = False
            if feature['regression'].regression is not None and feature['regression'].variance > 0:
                coefs.append(feature['regression'].regression.coef_[0][0])
                intercept.append(feature['regression'].regression.intercept_[0])
                zero_confidence.append(1)
            else:
                coefs.append(0)
                intercept.append(0)
                zero_confidence.append(0)
            Ns.append(feature['regression'].N if feature['regression'].N > feature[
                'regression'].effective_training_count else np.NaN)
            residuals.append(feature['regression'].residual if feature['regression'].residual is not None else np.NaN)
            meanX.append(feature['regression'].meanX if feature['regression'].meanX is not None else np.NaN)
            variance.append(feature['regression'].variance if feature['regression'].variance is not None else np.NaN)
        coefs = np.array(coefs)[col]
        intercept = np.array(intercept)[col]
        zero_confidence = np.array(zero_confidence)[col]
        residuals, Ns, meanX, variance = np.array(residuals)[col], np.array(Ns)[col], np.array(meanX)[col], \
                                         np.array(variance)[col]
        tvalue = float(delta) / (residuals * np.sqrt(1 + 1.0 / Ns + np.power(data - meanX, 2) / variance))
        confidence = np.ones_like(data)
        confidence[np.where(residuals > 0)] = (1 - t.sf(tvalue, (Ns - 2)) * 2)[np.where(residuals > 0)]
        confidence = confidence * zero_confidence
        evidential_support = (1 + confidence) / 2  # 正则化
        # 将计算出的evidential support写回
        csr_evidential_support = csr_matrix((evidential_support, (row, col)),
                                            shape=(len(self.variables), len(self.features)))
        for index, var in enumerate(self.variables):
            for feature_id in var['feature_set']:
                var['feature_set'][feature_id][0] = csr_evidential_support[index, feature_id]
        # 计算近似权重
        predict = data * coefs + intercept
        espredict = predict * evidential_support
        # espredict[np.where(polar_enforce == 0)] = np.minimum(espredict, 0)[np.where(polar_enforce == 0)]   #小于0就变成0
        # espredict[np.where(polar_enforce == 1)] = np.maximum(espredict, 0)[np.where(polar_enforce == 1)]   #大于0就变成1
        espredict = espredict * zero_confidence
        # 验证所有evidential support都在(0,1)之间
        assert len(np.where(evidential_support < 0)[0]) == 0
        assert len(np.where((1 - evidential_support) < 0)[0]) == 0
        loges = np.log(evidential_support)  # 取自然对数
        logunes = np.log(1 - evidential_support)
        evidential_support_logit = csr_matrix((loges, (row, col)), shape=(len(self.variables), len(self.features)))
        evidential_unsupport_logit = csr_matrix((logunes, (row, col)), shape=(len(self.variables), len(self.features)))
        p_es = np.exp(np.array(evidential_support_logit.sum(axis=1)))
        p_unes = np.exp(np.array(evidential_unsupport_logit.sum(axis=1)))
        approximate_weight = csr_matrix((espredict, (row, col)), shape=(len(self.variables), len(self.features)))
        approximate_weight = np.array(approximate_weight.sum(axis=1)).reshape(-1)
        # 将计算出来的近似权重写回variables
        for index, var in enumerate(self.variables):
            var['approximate_weight'] = approximate_weight[index]
        # 综合计算每个隐变量的evidential suppport,并写回self.variables
        for var_id in self.poential_variables_set:
            index = var_id
            var_p_es = p_es[index]
            var_p_unes = p_unes[index]
            self.variables[index]['evidential_support'] = float(var_p_es / (var_p_es + var_p_unes))
        logging.info("evidential_support_by_regression finished")


    def get_dict_rel_acc(self):
        '''
        get the accuracy of different types of relations
        :return:
        '''
        relations_name = set()
        relations_id = set()
        dict_rel_acc = dict()
        for feature in self.features:
            feature_id = feature['feature_id']
            if feature['feature_type'] == 'binary_feature':
                relations_id.add(feature_id)
                relations_name.add(feature['feature_name'])
        dict_reltype_edges = {rel: list() for rel in relations_name}
        for fid in relations_id:
            weight = self.features[fid]['weight']
            for (vid1,vid2) in weight:
                if self.variables[vid1]['is_evidence'] == True and self.variables[vid2]['is_evidence'] == True:
                    dict_reltype_edges[self.features[fid]['feature_name']].append((vid1,vid2))
        for rel,edges  in dict_reltype_edges.items():
            ture_rel_num = 0
            if len(edges) > 0:
                for (vid1,vid2) in edges:
                        if 'simi' in rel and self.variables[vid1]['label'] == self.variables[vid2]['label']:
                            ture_rel_num +=1
                        elif 'oppo' in rel and self.variables[vid1]['label'] != self.variables[vid2]['label']:
                            ture_rel_num +=1
                        dict_rel_acc[rel] = float(ture_rel_num/len(edges))
            else:
                dict_rel_acc[rel] = 0.9
        return dict_rel_acc

    def construct_mass_function_for_propensity(self,uncertain_degree, label_prob, unlabel_prob):
        '''
        # l: support for labeling
        # u: support for unalbeling
        '''
        return MassFunction({'l': (1 - uncertain_degree) * label_prob,
                             'u': (1 - uncertain_degree) * unlabel_prob,
                             'lu': uncertain_degree})


    def labeling_propensity_with_ds(self,mass_functions):
        combined_mass = gml_utils.combine_evidences_with_ds(mass_functions, normalization=True)
        return combined_mass

    def evidential_support_by_relation(self,variable_set,update_feature_set=None):
        '''
        计算给定隐变量集合中每个隐变量的evidential support,适用于ALSA
        :param update_feature_set:
        :param variable_set:
        :return:
        '''
        self.get_unlabeled_var_feature_evi()
        if type(variable_set) == list or type(variable_set) == set:
            mass_functions = list()
            for vid in variable_set:
                if len(self.variables[vid]['binary_feature_evi']) > 0:
                    for (anotherid, feature_name, feature_id) in self.variables[vid]['binary_feature_evi']:
                        # (anotherid,feature_name,feature_id)
                        rel_acc = self.dict_rel_acc[feature_name]
                        mass_functions.append(self.construct_mass_function_for_propensity(self.relation_evi_uncer_degree,rel_acc, 1-rel_acc))
                if len(self.variables[vid]['unary_feature_evi_prob']) > 0:
                    for (feature_id, feature_name, n_samples, neg_prob, pos_prob) in self.variables[vid]['unary_feature_evi_prob']:
                        # feature_id,feature_name,n_samples,neg_prob,pos_prob
                        mass_functions.append(self.construct_mass_function_for_propensity(self.word_evi_uncer_degree, max(pos_prob, neg_prob),min(pos_prob, neg_prob)))
                if len(mass_functions) > 0:
                    combine_evidential_support = self.labeling_propensity_with_ds(mass_functions)
                    self.variables[vid]['combine_evidential_support'] = combine_evidential_support
                    self.variables[vid]['evidential_support'] = combine_evidential_support['l']
                    mass_functions.clear()
                else:
                    self.variables[vid]['evidential_support'] = 0.0

            logging.info("evidential_support_by_relation finished")

    def evidential_support_by_custom(self,variable_set,update_feature_set=None):
        '''
        用户自定义计算evidential support
        :param update_feature_set:
        :return:
        '''
        pass

