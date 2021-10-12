import logging
import gml_utils
from pyds import MassFunction
import math

class ApproximateProbabilityEstimation:
    '''
    Approximate probability calculation
    '''
    def __init__(self, variables,features):
        self.variables = variables
        self.features = features
        self.word_evi_uncer_degree = 0.4
        self.relation_evi_uncer_degree = 0.1
        self.dict_rel_weight = self.init_binary_feature_weight(2,-2)


    def init_binary_feature_weight(self,value1,value2):   #value1 = 2, value2= -2
        '''
       #Set the initial value of the binary feature weight
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
        '''
        calculate evidence conflict
        @param mass_functions:
        @return:
        '''
        if len(mass_functions) == 0:
            conflict_degree = 0.0
        else: #combine Evidence support of multiple factors
            combined_mass = gml_utils.combine_evidences_with_ds(mass_functions, normalization=True)
            if combined_mass['p'] != 0:
                conflict_degree =  combined_mass['p']
            else:
                conflict_degree =  combined_mass['n']
        return conflict_degree

    def get_pos_prob_based_relation(self, var_id, weight):
        '''
        @param var_id:
        @param weight:
        @return:
        '''
        if self.variables[var_id]['label'] == 1:
            pos_prob = math.exp(weight) / (1 + math.exp(weight))
        elif self.variables[var_id]['label'] == 0:
            pos_prob = 1 / (1 + math.exp(weight))
        else:
            raise ValueError('The relation does not contain labeled  .')
        return pos_prob

    def construct_mass_function_for_confict(self, uncertain_degree, pos_prob, neg_prob):
        '''
        # p: positive
        # n: negative
        @param uncertain_degree:
        @param pos_prob:
        @param neg_prob:
        @return:
        '''
        return MassFunction({'p': (1 - uncertain_degree) * pos_prob,
                             'n': (1 - uncertain_degree) * neg_prob,
                             'pn': uncertain_degree})

    def construct_mass_function_for_ER(self, tau, alpha, confidence,featureValue):
        '''
        # p: positive
        # n: negative
        @param tau:
        @param alpha:
        @param confidence:
        @param featureValue:
        @return:
        '''
        return MassFunction({'p': 1/(1+math.exp(-confidence*(tau*featureValue+alpha))),  #Probability of being the same entity
                             'n': 1/(1+math.exp(confidence*(tau*featureValue+alpha))),  #Probability of not being the same entity
                             })

    def approximate_probability_estimation(self, variable_set):
        '''
        Calculate the evidence support for each hidden variable
        @param variable_set:
        @return:
        '''
        if type(variable_set) == list or type(variable_set) == set:
            mass_functions = list()
            for id in variable_set: #Choose a different mass function according to the factor type to calculate Approximate probability 
                for fid in self.variables[id]['feature_set']:
                    if self.features[fid]['feature_type']== 'unary_feature': #If it is a unary factor, it is necessary to judge whether there is a feature value or whether it needs to be parameterized
                        if self.features[fid]['parameterize']  == 1 :
                            if self.features[fid]['regression'].regression is not None and self.features[fid]['regression'].variance > 0 :
                                mass_functions.append(self.construct_mass_function_for_ER(self.features[fid]['regression'].regression.coef_[0][0],self.features[fid]['regression'].regression.intercept_[0],self.variables[id]['feature_set'][fid][0],self.variables[id]['feature_set'][fid][1]))
                        else:
                            if len(self.variables[id]['unary_feature_evi_prob']) > 0:
                                for (feature_id,feature_name,n_samples,neg_prob,pos_prob) in self.variables[id]['unary_feature_evi_prob']:
                                    mass_functions.append(self.construct_mass_function_for_confict(self.word_evi_uncer_degree, pos_prob, neg_prob))
                    if self.features[fid]['feature_type']== 'binary_feature': 
                        if len(self.variables[id]['binary_feature_evi']) > 0:
                            for (anotherid,feature_name,feature_id) in self.variables[id]['binary_feature_evi']:
                                if fid == feature_id:
                                    pos_prob = self.get_pos_prob_based_relation(anotherid, self.dict_rel_weight[feature_name])
                                    mass_functions.append(self.construct_mass_function_for_confict(self.relation_evi_uncer_degree, pos_prob, 1 - pos_prob))
                if len(mass_functions) > 0: #Write the final ApproximateProbability
                    conflict = self.labeling_conflict_with_ds(mass_functions)
                    self.variables[id]['approximate_probability'] = conflict
                    self.variables[id]['entropy'] = gml_utils.entropy(conflict)
                    mass_functions.clear()
                else:
                    self.variables[id]['approximate_probability'] = 0
                    self.variables[id]['entropy'] = 0.99
        else:
            raise ValueError('input type error')
        logging.info("approximate_probability estimation finished")
