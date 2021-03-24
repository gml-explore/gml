import random
from collections import namedtuple
import logging
from numbskull_extend.numbskulltypes import *

class ConstructSubgraph:

    def __init__(self, variables, features,balance):
        self.variables = variables
        self.features = features
        self.balance = balance      #if need to balance the number of 0 and 1 in the evidence variable

    def construct_subgraph(self, evidences,var_id):
        connected_var_set, connected_edge_set, connected_feature_set = evidences
        var_map = dict()  # Used to record the mapping between self.variables and numbskull variable id -(self,numbskull)
        # 1.Initialize variable
        var_num = len(connected_var_set)
        variable = np.zeros(var_num, Variable)
        variable_index = 0
        for id in connected_var_set:
            variable[variable_index]["isEvidence"] = self.variables[id]['is_evidence']  # The evidence variable is True, otherwise it is False
            variable[variable_index]["initialValue"] = self.variables[id]['label']  # The initial value of the variable
            variable[variable_index]["dataType"] = 0  # datatype=0 means it is a bool variable, when it is 1, it means a non-Boolean variable.
            variable[variable_index]["cardinality"] = 2  #Two categories
            var_map[id] = variable_index
            variable_index += 1
        #2.Initialize weight: The weight in numbskull corresponds to the external feature, and multiple factors can share the same weight
        weight = np.zeros(len(connected_feature_set), Weight)  # The number of weights is equal to the number of features
        alpha_bound = np.zeros(len(connected_feature_set), AlphaBound)
        tau_bound = np.zeros(len(connected_feature_set), TauBound)
        feature_map_weight = dict()  # Record the mapping between feature id and weight id  [feature_id,weight_id]
        weight_map_feature = dict()  # Used to record the mapping between weight id and feature id  [weight_id,feature_id]
        weight_map_factor = dict()  # Used to record the factor owned by each weight
        weight_index = 0
        for feature_id in connected_feature_set:
            weight[weight_index]["isFixed"] = False  # When learning the factor graph, the weight value is fixed to False
            weight[weight_index]["parameterize"] = self.features[feature_id]['parameterize']  #The weight is whether it needs to be parameterized
            #If you need to parameterize, you need to initialize the parameter value and upper and lower bounds
            if self.features[feature_id]['parameterize'] == 1:
                weight[weight_index]["a"] = self.features[feature_id]['tau']
                weight[weight_index]["b"] = self.features[feature_id]['alpha']
                alpha_bound[weight_index]['lowerBound'] = self.features[feature_id]['alpha_bound'][0]
                alpha_bound[weight_index]['upperBound'] = self.features[feature_id]['alpha_bound'][1]
                tau_bound[weight_index]['lowerBound'] = self.features[feature_id]['tau_bound'][0]
                tau_bound[weight_index]['upperBound'] = self.features[feature_id]['tau_bound'][1]
            key = (random.sample(self.features[feature_id]['weight'].keys(), 1))[0]
            weight[weight_index]["initialValue"] = self.features[feature_id]['weight'][key][0]  # Here a weight may have many weight_values, and the assignment is the first
            feature_map_weight[feature_id] = weight_index
            weight_map_feature[weight_index] = feature_id
            weight_map_factor[weight_index] = set()
            weight_index += 1
        #3.Initialize factor,fmap,edges
        binary_feature_edge = list()  # Set of binary-factor edges
        unary_feature_edge = list()   #Set of single-factor edges
        for elem in connected_edge_set:   #elem: [feature_id,(varid1,varid2)] or  [feature_id,varid]
            if self.features[elem[0]]['feature_type'] == 'unary_feature':
                unary_feature_edge.append(elem)
            elif self.features[elem[0]]['feature_type'] == 'binary_feature':
                binary_feature_edge.append(elem)
        edges_num = len(unary_feature_edge) + 2 * len(binary_feature_edge)  # 边的数目=单因子数目+2*双因子数目
        factor = np.zeros(len(unary_feature_edge) + len(binary_feature_edge), Factor)  # Number of factors = number of single factors + number of binary factors
        fmap = np.zeros(edges_num, FactorToVar)  # factor[factor_index]->fmp_index，fmap[fmp_index]->var_index
        domain_mask = np.zeros(var_num, np.bool)
        edges = list()
        unary_edge = namedtuple('unary_edge', ['index', 'factorId', 'varId'])  # unary factor edge
        binary_edge = namedtuple('binary_edge', ['index', 'factorId', 'varId1', 'varId2'])  # binary facotr edge
        factor_index = 0
        fmp_index = 0
        edge_index = 0
        # Initialize single factor, single factor temporarily use factor function 13
        for elem in unary_feature_edge:  # [feature_id,var_id]
            feature_id = elem[0]
            vid = elem[1]  #The variable id on the side is used to find feature_value later
            factor[factor_index]["factorFunction"] = 13
            factor[factor_index]["weightId"] = feature_map_weight[feature_id]  #The corresponding weight id of the factor
            factor[factor_index]["featureValue"] = 1  # factor featureValue,unused
            factor[factor_index]["arity"] = 1  # Single factor degree is 1
            factor[factor_index]["ftv_offset"] = fmp_index  # The single factor offset is increased by 1 each time
            fmap[fmp_index]["vid"] = var_map[vid]
            if self.features[feature_id]['parameterize'] == 1:
                fmap[fmp_index]["x"] = self.variables[vid]['feature_set'][feature_id][1]  # feature_value
                fmap[fmp_index]["theta"] = self.variables[vid]['feature_set'][feature_id][0]  # theta
            weight_map_factor[feature_map_weight[elem[0]]].add(factor_index) #Record which weight this factor belongs to
            edges.append(unary_edge(edge_index, factor_index, var_map[vid]))
            fmp_index += 1
            factor_index += 1
            edge_index += 1
        # Initialize double factor, double factor temporarily use factor function 9
        for elem in binary_feature_edge:  # [feature_id,（var_id1，var_id2)]
            feature_id = elem[0]
            vid = elem[1]
            factor[factor_index]["factorFunction"] = 9
            factor[factor_index]["weightId"] = feature_map_weight[feature_id]
            factor[factor_index]["featureValue"] = 1  #The weight used to zoom in or zoom out the factor, the default is 1, which means neither zoom in nor zoom out
            factor[factor_index]["arity"] = 2  # binary factor degree is 2
            factor[factor_index]["ftv_offset"] = fmp_index  #Increase the offset by 2 each time
            weight_map_factor[feature_map_weight[feature_id]].add(factor_index)
            edges.append(binary_edge(edge_index, factor_index, var_map[vid[0]], var_map[vid[1]]))
            for id in vid:
                fmap[fmp_index]["vid"] = var_map[id]
                # To support subsequent binary-factor functionalization
                if self.features[feature_id]['parameterize'] == 1:
                    fmap[fmp_index]["x"] = self.variables[vid]['feature_set'][feature_id][1]  # feature_value
                    fmap[fmp_index]["theta"] = self.variables[vid]['feature_set'][feature_id][0]  # theta
                fmp_index += 1
            factor_index += 1
            edge_index += 1
        # Balance: Expand the number of variables on the lesser side
        if self.balance:
            #生成sampleList用于平衡化
            label0_var = list()
            label1_var = list()
            poential_var = list()  #There is more than one hidden variable in the subgraph
            for id in range(0,var_num):
                if variable[id]['isEvidence'] == True and variable[id]['initialValue'] == 0:
                    label0_var.append(id)
                elif variable[id]['isEvidence'] == True and variable[id]['initialValue'] == 1:
                    label1_var.append(id)
                #Hidden variables do not participate in balancing
                elif variable[id]['isEvidence'] == False and id != var_map[var_id]:
                    poential_var.append(id)
            if len(label0_var)>=len(label1_var):
                extend_1 = True
                diff = len(label0_var)-len(label1_var)
            else:
                extend_1 = False
                diff = len(label1_var) - len(label0_var)
            #sample_num = len(label1_var) + len(label0_var) + diff + 1+len(poential_var)
            sample_num = len(connected_var_set)+diff
            index_list = [x for x in range(0,sample_num-1)]
            random.shuffle(index_list)  #In order to disrupt the 0-1 order
            sample_list = np.zeros(sample_num, SampleList)
            sample_index = 0
            #Add all 0
            for id in label0_var:
                sample_list[index_list[sample_index]]['vid'] = id
                sample_index += 1
            #Add all 1
            for id in label1_var:
                sample_list[index_list[sample_index]]['vid'] = id
                sample_index += 1
            #Expand the difference
            if diff>0:
                #expand 1
                if extend_1:
                   for i in range(0,diff):
                       sample_list[index_list[sample_index]]['vid'] = random.choice(label1_var)
                       sample_index += 1
                #expand 0
                elif not extend_1:
                    for i in range(0, diff):
                        sample_list[index_list[sample_index]]['vid'] = random.choice(label0_var)
                        sample_index += 1
            #Add the original hidden variables (except for the target hidden variables)
            for id in enumerate(poential_var):
                sample_list[index_list[sample_index]]['vid'] = id
                sample_index += 1
            #The target latent variable is added at the end
            sample_list[sample_num-1]['vid'] = var_map[var_id]
        else:
            sample_list = None
        # Initialize wmap (WeightToFactor)) and wfactor (FactorToWeight) for batch gradient descent
        wmap = np.zeros(len(weight), WeightToFactor)  #Used to find all factors associated with each weight
        wfactor = np.zeros(len(factor), FactorToWeight)  # Factors organized in order of weight
        wfactor_index = 0
        for weightId, factorSet in weight_map_factor.items():
            count = 0
            wmap[weightId]["weightId"] = weightId
            wmap[weightId]["weight_index_offset"] = wfactor_index
            for factorId in factorSet:
                wfactor[wfactor_index]["factorId"] = factorId
                count += 1
                wfactor_index += 1
            wmap[weightId]["weight_index_length"] = count
        logging.info("construct subgraph  finished")
        return weight, variable, factor, fmap, domain_mask, edges_num, var_map, alpha_bound, tau_bound, weight_map_feature, sample_list, wmap, wfactor

