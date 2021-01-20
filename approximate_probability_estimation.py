import logging

import gml_utils
from pyds import MassFunction
import math


# 近似概率计算的一些方法
class ApproximateProbabilityEstimation:
    def __init__(self, variables,features,method='relation'):
        self.variables = variables
        self.features = features
        self.word_evi_uncer_degree = 0.4
        self.relation_evi_uncer_degree = 0.1
        if method == 'relation':
            self.dict_rel_weight = self.init_binary_feature_weight(2,-2)

    def init_binary_feature_weight(self,value1,value2):   #value1 = 2, value2= -2
        '''
         #设置binary feature权重的初始值
        :param value1:
        :param value2:
        :return:
        '''
        dict_rel_weight = dict()
        dict_rel_weight['asp2asp_sequence_simi'] = value1
        dict_rel_weight['asp2asp_intrasent_simi'] = value1
        dict_rel_weight['asp2asp_sequence_oppo'] = value2
        dict_rel_weight['asp2asp_intrasent_oppo'] = value2
        return dict_rel_weight

    def labeling_conflict_with_ds(self, mass_functions):
        '''情感分析中使用，用于计算证据冲突'''
        if len(mass_functions) < 2:
            conflict_degree = 0.0
        else:
            combined_mass = gml_utils.combine_evidences_with_ds(mass_functions, normalization=True)
            unnormalize_combined_mass = gml_utils.combine_evidences_with_ds(mass_functions, normalization=False)
            if combined_mass['p'] != 0:
                conflict_degree = 1 - unnormalize_combined_mass['p'] / combined_mass['p']
            else:
                conflict_degree = 1 - unnormalize_combined_mass['n'] / combined_mass['n']
        return conflict_degree

    def get_pos_prob_based_relation(self, var_id, weight):
        '''情感分析中使用，用于计算近似概率'''
        if self.variables[var_id]['label'] == 1:
            pos_prob = math.exp(weight) / (1 + math.exp(weight))
        elif self.variables[var_id]['label'] == 0:
            pos_prob = 1 / (1 + math.exp(weight))
        else:
            print(var_id)
            raise ValueError('The relation does not contain labeled  .')
        return pos_prob

    def construct_mass_function_for_confict(self, uncertain_degree, pos_prob, neg_prob):
        '''
        情感分析中使用，用于计算证据冲突，最终排序选topk时使用的是 L中的值
        # l: support for labeling
        # u: support for unalbeling
        '''
        return MassFunction({'p': (1 - uncertain_degree) * pos_prob,
                             'n': (1 - uncertain_degree) * neg_prob,
                             'pn': uncertain_degree})

    def approximate_probability_estimation_by_interval(self, variable_set):
        '''
        计算给定隐变量集合中每个隐变量的近似概率和熵，用于选topk,适用于ER
        :param variable_set:
        :return:
        '''
        if type(variable_set) == set or type(variable_set) == list:
            for id in variable_set:
                self.variables[id]['approximate_probability'] = gml_utils.open_p(self.variables[id]['approximate_weight'])
                self.variables[id]['entropy'] = gml_utils.entropy(self.variables[id]['approximate_probability'])
        else:
            raise ValueError('input type error')
        logging.info("approximate_probability_estimation_by_interval finished")

    def approximate_probability_estimation_by_relation(self, variable_set):
        '''
        计算给定隐变量集合中每个隐变量的近似概率和熵，用于选topk,适用于ALSA
        :param variable_set:
        :return:
        '''
        if type(variable_set) == list or type(variable_set) == set:
            mass_functions = list()
            for id in variable_set:
                if len(self.variables[id]['binary_feature_evi']) > 0:
                    for (anotherid,feature_name,feature_id) in self.variables[id]['binary_feature_evi']:
                        #(anotherid,feature_name,feature_id)
                        pos_prob = self.get_pos_prob_based_relation(anotherid, self.dict_rel_weight[feature_name])
                        mass_functions.append(self.construct_mass_function_for_confict(self.relation_evi_uncer_degree, pos_prob, 1 - pos_prob))
                if len(self.variables[id]['unary_feature_evi_prob']) > 0:
                    for (feature_id,feature_name,n_samples,neg_prob,pos_prob) in self.variables[id]['unary_feature_evi_prob']:
                        #feature_id,feature_name,n_samples,neg_prob,pos_prob
                        mass_functions.append(self.construct_mass_function_for_confict(self.word_evi_uncer_degree, pos_prob, neg_prob))
                if len(mass_functions) > 0:
                    conflict = self.labeling_conflict_with_ds(mass_functions)
                    self.variables[id]['approximate_probability'] = conflict
                    self.variables[id]['entropy'] = conflict
                    mass_functions.clear()
                else:
                    self.variables[id]['approximate_probability'] = 0
                    self.variables[id]['entropy'] = 0.99
        else:
            raise ValueError('input type error')
            logging.info("approximate_probability_estimation_by_relation finished")

    def approximate_probability_estimation_by_custom(self, variable_set):
        '''
        用户自定义的计算近似概率的方法
        :param variable_set:
        :return:
        '''
        pass
