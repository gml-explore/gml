import math
import numpy as np
import pandas as pd
from copy import  copy

def load_easy_instance_from_file(filename):
    '''
    从csv文件中加载easy
    '''

    easy_data = pd.read_csv(filename)
    easy_pair = {'var_id': 0, 'label': 1}
    easy_pair_list = []
    for i in range(len(easy_data)):
        easy_pair['var_id'] = easy_data['id'][i]
        easy_pair['label'] = easy_data['label'][i]
        easy_pair_list.append(copy(easy_pair))
    return easy_pair_list

def separate_variables(variables):
    '''
    将variables分成证据变量和隐变量
    '''
    observed_variables_set = set()
    poential_variables_set = set()
    for variable in variables:
        if variable['is_evidence'] == True:
            observed_variables_set.add(variable['var_id'])
        else:
            poential_variables_set.add(variable['var_id'])
    return observed_variables_set,poential_variables_set

def init_evidence_interval(evidence_interval_count):
    '''初始化证据区间
    输出：一个包含evidence_interval_count个区间的list
    '''
    evidence_interval = list()
    step = float(1) / evidence_interval_count
    previousleft = None
    previousright = 0
    for intervalindex in range(0, evidence_interval_count):
        currentleft = previousright
        currentright = currentleft + step
        if intervalindex == evidence_interval_count - 1:
            currentright = 1 + 1e-3
        previousleft = currentleft
        previousright = currentright
        evidence_interval.append([currentleft, currentright])
    return evidence_interval

def init_evidence(features,evidence_interval,observed_variables_set):
    '''
    初始化所有feature的evidence_interval属性和evidence_count属性
    :return:
    '''
    for feature in features:
        evidence_count = 0
        intervals = [set(), set(), set(), set(), set(), set(), set(), set(), set(), set()]
        if feature['parameterize'] == 1:
            weight = feature['weight']
            feature['evidence_interval'] = intervals
            for kv in weight.items():
                if kv[0] in observed_variables_set:
                    for interval_index in range(0, len(evidence_interval)):
                        if kv[1][1] >= evidence_interval[interval_index][0] and kv[1][1] < \
                                evidence_interval[interval_index][1]:
                            feature['evidence_interval'][interval_index].add(kv[0])
                            evidence_count += 1
            feature['evidence_count'] = evidence_count

def write_labeled_var_to_evidence_interval(variables,features,var_id,evidence_interval):
    '''
    因为每个feature维护了evidence_interval属性，所以每标记一个变量之后，需要更新这个属性
    :param var_id:
    :return:
    '''
    var_index = var_id
    feature_set = variables[var_index]['feature_set']
    for kv in feature_set.items():
        for interval_index in range(0, len(evidence_interval)):
            if kv[1][1] >= evidence_interval[interval_index][0] and kv[1][1] < \
                    evidence_interval[interval_index][1]:
                features[kv[0]]['evidence_interval'][interval_index].add(var_id)
                features[kv[0]]['evidence_count'] += 1

def init_bound(variables,features):
    '''
    在标记完Easy之后，初始化每个feature的alpha的bound和tau的bound
    @param variables:
    @param features:
    @return:
    '''
    for feature in features:
        if feature['parameterize'] == 1:
            feature_evidence0_count = 0
            feature_evidence1_count = 0
            feature_evidence0_sum = 0
            feature_evidence1_sum = 0
            for vid in feature['weight'].keys():
                if variables[vid]['is_evidence'] == True:
                    if variables[vid]['label'] == 0:
                        feature_evidence0_count += 1
                        feature_evidence0_sum +=  feature['weight'][vid][1]
                    elif variables[vid]['label'] == 1:
                        feature_evidence1_count += 1
                        feature_evidence1_sum += feature['weight'][vid][1]
            if feature_evidence0_count!=0:
                bound0 = feature_evidence0_sum/feature_evidence0_count
            else:
                bound0 = 0
            if feature_evidence1_count != 0:
                bound1 = feature_evidence1_sum/feature_evidence1_count
            else:
                bound1 = 0
            feature['alpha_bound'] = copy([bound0,bound1])
            feature['tau_bound'] = copy([0,10])

def entropy(probability):
    '''给定概率之后计算熵
    输入：
    probability ： 单个概率或者概率列表
    输出： 单个熵或者熵的列表
    '''
    if type(probability) == np.float64 or type(probability) == np.float32 or type(probability) == float or type(
            probability) == int:
        if math.isinf(probability) == True:
            return probability
        else:
            if probability <= 0 or probability >= 1:
                return 0
            else:
                return 0 - (probability * math.log(probability, 2) + (1 - probability) * math.log((1 - probability),
                                                                                                  2))
    else:
        if type(probability) == list:
            entropy_list = []
            for each_probability in probability:
                entropy_list.append(entropy(each_probability))
            return entropy_list
        else:
            return None

def open_p(weight):
    return float(1) / float(1 + math.exp(- weight))

def combine_evidences_with_ds(mass_functions, normalization):
    # combine evidences from different sources
    if len(mass_functions) < 2:
        combined_mass = mass_functions[0]
    else:
        combined_mass = mass_functions[0].combine_conjunctive(mass_functions[1], normalization)

        if len(mass_functions) > 2:
            for mass_func in mass_functions[2: len(mass_functions)]:
                combined_mass = combined_mass.combine_conjunctive(mass_func, normalization)
    return combined_mass
