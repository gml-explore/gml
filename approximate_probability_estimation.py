import logging
import gml_utils
from pyds import MassFunction
import math


# 近似概率计算的一些方法
class ApproximateProbabilityEstimation:
    def __init__(self, variables,features):
        self.variables = variables
        self.features = features
        self.word_evi_uncer_degree = 0.4
        self.relation_evi_uncer_degree = 0.1
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
            #unnormalize_combined_mass = gml_utils.combine_evidences_with_ds(mass_functions, normalization=False)
            if combined_mass['p'] != 0:
                conflict_degree =  combined_mass['p']
            else:
                conflict_degree =  combined_mass['n']
        return conflict_degree

    def get_pos_prob_based_relation(self, var_id, weight):
        '''情感分析中使用，用于计算近似概率'''
        if self.variables[var_id]['label'] == 1:
            pos_prob = math.exp(weight) / (1 + math.exp(weight))
        elif self.variables[var_id]['label'] == 0:
            pos_prob = 1 / (1 + math.exp(weight))
        else:
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

    def construct_mass_function_for_ER(self, tau, alpha, confidence,featureValue):
        '''
        实体识别中使用，用于计算证据冲突，最终排序选topk时使用的是 L中的值
        # l: support for labeling
        # u: support for unalbeling
        '''
        return MassFunction({'p': 1/(1+math.exp(-confidence*(tau*featureValue+alpha))),  #是同一实体的概率
                             'n': 1/(1+math.exp(confidence*(tau*featureValue+alpha))),  #不是同一实体的概率
                             })

    def approximate_probability_estimation(self, variable_set):
        '''
        计算给定隐变量集合中每个隐变量的近似概率和熵，用于选topk,适用于ER
        :param variable_set:
        :return:
        '''
        if type(variable_set) == list or type(variable_set) == set:
            mass_functions = list()
            for id in variable_set:
                for fid in self.variables[id]['feature_set']:
                    if self.features[fid]['feature_type']== 'unary_feature': #如果是单因子 还需判断是否有featurevalue 或判断是否需要参数化
                        if self.features[fid]['parameterize']  == 1 : #如果需要参数化
                            if self.features[fid]['regression'].regression is not None and self.features[fid]['regression'].variance > 0 :
                                mass_functions.append(self.construct_mass_function_for_ER(self.features[fid]['regression'].regression.coef_[0][0],self.features[fid]['regression'].regression.intercept_[0],self.variables[id]['feature_set'][fid][0],self.variables[id]['feature_set'][fid][1]))
                                print(self.features[fid]['regression'].regression.coef_[0][0],self.features[fid]['regression'].regression.intercept_[0],self.variables[id]['feature_set'][fid][0],self.variables[id]['feature_set'][fid][1])
                        else:
                            if len(self.variables[id]['unary_feature_evi_prob']) > 0:
                                for (feature_id,feature_name,n_samples,neg_prob,pos_prob) in self.variables[id]['unary_feature_evi_prob']:
                                     #feature_id,feature_name,n_samples,neg_prob,pos_prob
                                    mass_functions.append(self.construct_mass_function_for_confict(self.word_evi_uncer_degree, pos_prob, neg_prob))
                    if self.features[fid]['feature_type']== 'binary_feature': 
                        if len(self.variables[id]['binary_feature_evi']) > 0:
                            for (anotherid,feature_name,feature_id) in self.variables[id]['binary_feature_evi']:
                        #(anotherid,feature_name,feature_id)
                                pos_prob = self.get_pos_prob_based_relation(anotherid, self.dict_rel_weight[feature_name])
                                mass_functions.append(self.construct_mass_function_for_confict(self.relation_evi_uncer_degree, pos_prob, 1 - pos_prob))
                if len(mass_functions) > 0:
                    conflict = self.labeling_conflict_with_ds(mass_functions)
                    self.variables[id]['approximate_probability'] = conflict
                    self.variables[id]['entropy'] = gml_utils.entropy(conflict)
                    mass_functions.clear()
                else:
                    self.variables[id]['approximate_probability'] = 0
                    self.variables[id]['entropy'] = 0.99
        else:
            raise ValueError('input type error')
        logging.info("approximate_probability_estimation finished")
