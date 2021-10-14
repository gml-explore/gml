#提供挑选证据的一些方法
import logging
import random
from copy import copy


class EvidenceSelect:
    def __init__(self, variables, features, interval_evidence_limit = 200, subgraph_max_num=3000,each_feature_evidence_limit = 2000):
        self.variables = variables
        self.features = features
        self.subgraph_max_num = subgraph_max_num    #Maximum number of variables allowed in the subgraph
        self.interval_evidence_limit = interval_evidence_limit   #Uniformly divided into 10 intervals, the number of evidence variables sampled in each interval
        self.each_feature_evidence_limit = each_feature_evidence_limit   #Limit the number of evidence variables for each single factor in the subgraph

    def evidence_select(self, var_id):
        '''
        Uniform evidence selection method
        @param var_id:
        @return:
        connected_var_set :  Subgraph variable set
        connected_edge_set:  Subgraph egde set
        connected_feature_set: Subgraph feature set
        '''
        if type(var_id) == int:
            subgraph_max_num = self.subgraph_max_num
            random_sample_num = self.each_feature_evidence_limit  # The number of evidences to be sampled for each single factor when there is no featureValue
            connected_var_set = set()  # Finally add the hidden variable id
            connected_edge_set = set()  # [feature_id,var_id]  or [feature_id,(id1,id2)]
            connected_feature_set = set()
            feature_set = self.variables[var_id]['feature_set']
            binary_feature_set = set()
            unary_feature_set = set()
            current_var_set = set() #The basic variables of the current hop
            current_var_set.add(var_id)
            next_var_set = set()
            k_hop = 2
            # Divide the double factor and single factor of this hidden variable
            for feature_id in feature_set.keys():
                if self.features[feature_id]['feature_type'] == 'binary_feature':
                    binary_feature_set.add(feature_id)
                elif self.features[feature_id]['feature_type'] == 'unary_feature':
                    unary_feature_set.add(feature_id)
            #Deal with double factors and find evidence variables for k-hop jumps
            # for k in range(k_hop):
            #     # Each round adds the adjacent evidence variable of the previous round of hidden variables
            #     for varid in current_var_set:
            #         feature_set = self.variables[varid]['feature_set']
            #         for feature_id in feature_set.keys():
            #             # If this latent variable id is contained in two variable ids connected by a two-factor, and the other variable is an evidence variable,
            #             # then this evidence variable is added to the next round of next_var_set, and the relevant features and edges are counted
            #             if self.features[feature_id]['feature_type'] == 'binary_feature':
            #                 weight = self.features[feature_id]['weight']
            #                 for id in weight.keys():
            #                     if type(id) == tuple and varid in id:
            #                         another_var_id = id[0] if id[0] != varid else id[1]
            #                         if self.variables[another_var_id]['is_evidence'] == True:
            #                             next_var_set.add(another_var_id)
            #                             connected_feature_set.add(feature_id)
            #                             connected_edge_set.add((feature_id, id))   #[feature_id,(id1,id2)]
            #         connected_var_set = connected_var_set.union(next_var_set)
            #         current_var_set = next_var_set
            #         next_var_set.clear()
            #Deal with double factors, find the evidence in the first round, and then find the evidence variables on the double factors connected by the latent variables in the first round
            for k in range(k_hop):
                # Each round adds the adjacent evidence variable of the previous round of hidden variables
                tmp_dict = {}
                for varid in current_var_set:
                    feature_set = self.variables[varid]['feature_set']
                    for feature_id in feature_set.keys():
                        # If this latent variable id is contained in two variable ids connected by a two-factor, and the other variable is an evidence variable,
                        # then this evidence variable is added to the next round of next_var_set, and the relevant features and edges are counted
                        if self.features[feature_id]['feature_type'] == 'binary_feature':
                            weight = self.features[feature_id]['weight']
                            for id in weight.keys():
                                if type(id) == tuple and varid in id:
                                    another_var_id = id[0] if id[0] != varid else id[1]
                                    if self.variables[another_var_id]['is_evidence'] == True:
                                        connected_feature_set.add(feature_id)
                                        connected_edge_set.add((feature_id, id))   #[feature_id,(id1,id2)]
                                        connected_var_set.add(another_var_id)
                                        if (var_id,varid) in tmp_dict:
                                            tmp_feature =tmp_dict[(var_id,varid)]
                                            connected_edge_set.add((tmp_feature,(var_id,varid)))
                                        elif (varid,var_id) in tmp_dict:
                                            tmp_feature = tmp_dict[(varid,var_id)]
                                            connected_edge_set.add((tmp_feature, (varid, var_id)))
                                    elif self.variables[another_var_id]['is_evidence'] == False:
                                        next_var_set.add(another_var_id)
                                        tmp_dict[id] = feature_id
                    connected_var_set = connected_var_set.union(current_var_set)
                current_var_set.clear()
                current_var_set = copy(next_var_set)
                # current_var_set = next_var_set
                next_var_set.clear()
            # Deal with single factor: limit the number of evidence variables for each single factor, and sample when it exceeds
            subgraph_capacity = subgraph_max_num - len(connected_var_set) - 1  # Maximum number of variables that can be added
            unary_evidence_set = set()
            unary_potential_set = set()
            # First judge whether all single-factor evidence is added to whether the maximum limit is exceeded
            for feature_id in unary_feature_set:
                weight = self.features[feature_id]['weight']
                for vid in weight.keys():
                    if self.variables[vid]['is_evidence'] == True:
                        unary_evidence_set.add(vid)
                    else:
                        unary_potential_set.add(vid)
            # If the maximum capacity of the sub-picture is not exceeded, add all
            if len(unary_evidence_set) <= subgraph_capacity:
                for feature_id in unary_feature_set:
                    weight = self.features[feature_id]['weight']
                    for vid in weight.keys():
                        if self.variables[vid]['is_evidence'] == True:
                            connected_var_set.add(vid)
                            connected_feature_set.add(feature_id)
                            connected_edge_set.add((feature_id, vid))  # [feature_id,id1]
            # If it exceeds, limit the number of evidences for each single factor, and sample according to whether there is feature_value
            if len(unary_evidence_set) > subgraph_capacity:
                for feature_id in unary_feature_set:
                    # Sampling by interval with feature_value
                    if self.features[feature_id]['parameterize'] == 1:
                        if self.features[feature_id]['evidence_count'] > 0 and (self.features[feature_id]['monotonicity'] == True and self.features[feature_id]['alpha_bound'][0] < self.features[feature_id]['alpha_bound'][1]):
                            connected_feature_set.add(feature_id)
                            evidence_interval = self.features[feature_id]['evidence_interval']
                            for interval in evidence_interval:
                                # If the evidence variable in this interval is less than 200, add them all
                                if len(interval) <= self.interval_evidence_limit:
                                    connected_var_set = connected_var_set.union(interval)
                                    for vid in interval:
                                        connected_edge_set.add((feature_id, vid))
                                else:
                                    # If it is greater than 200, randomly sample 200
                                    sample = random.sample(list(interval), self.interval_evidence_limit)
                                    connected_var_set = connected_var_set.union(sample)
                                    for vid in sample:
                                        connected_edge_set.add((feature_id, vid))     #(feature_id,v_id)
                    # Random sampling without feature_value
                    if self.features[feature_id]['parameterize'] == 0 and self.features[feature_id]['evidence_count'] > 0:
                        connected_feature_set.add(feature_id)
                        weight = self.features[feature_id]['weight']
                        unary_feature_evidence = set()  #The set of all evidence variables connected on this feature
                        for vid in weight.keys():
                            if self.variables[vid]['is_evidence'] == True:
                                unary_feature_evidence.add(vid)
                        sample = random.sample(list(unary_feature_evidence), random_sample_num)
                        connected_var_set = connected_var_set.union(sample)
                        unary_feature_evidence.clear()
                        for vid in sample:
                            connected_edge_set.add((feature_id, vid))
            #If the subgraph is too small, add the hidden variable on the single factor (only for the unparameterized single factor)
            subgraph_capacity = subgraph_max_num - len(connected_var_set) - 1
            unary_connected_unlabeled_var = list()
            unary_connected_unlabeled_edge = list()
            unary_connected_unlabeled_feature = list()
            feature_set = self.variables[var_id]['feature_set']
            for feature_id in feature_set.keys():
                if self.features[feature_id]['feature_type'] == 'unary_feature' and self.features[feature_id]['parameterize'] == 0:
                    weight = self.features[feature_id]['weight']
                    for id in weight.keys():
                        # First find the set of evidence variables and hidden variables, including related edges and features
                        if self.variables[id]['is_evidence'] == False:
                            unary_connected_unlabeled_var.append(id)
                            unary_connected_unlabeled_feature.append(feature_id)
                            unary_connected_unlabeled_edge.append((feature_id, id))
            if (subgraph_capacity > len(unary_connected_unlabeled_var)):
                connected_var_set = connected_var_set.union(set(unary_connected_unlabeled_var))
                connected_feature_set = connected_feature_set.union((set(unary_connected_unlabeled_feature)))
                connected_edge_set = connected_edge_set.union(set(unary_connected_unlabeled_edge))
            else:
                connected_var_set = connected_var_set.union(
                    set(unary_connected_unlabeled_var[:subgraph_capacity - len(unary_connected_unlabeled_var)]))
                connected_feature_set = connected_feature_set.union(
                    set(unary_connected_unlabeled_feature[:subgraph_capacity - len(unary_connected_unlabeled_var)]))
                connected_edge_set = connected_edge_set.union(
                    set(unary_connected_unlabeled_edge[:subgraph_capacity - len(unary_connected_unlabeled_var)]))
            # Add target latent variable related structure
            connected_var_set = list(connected_var_set)
            connected_edge_set = list(connected_edge_set)
            # connected_var_set.append(var_id)
            for feature_id in connected_feature_set:
                if self.features[feature_id]['feature_type'] == 'unary_feature':
                    connected_edge_set.append((feature_id, var_id))
            logging.info("select evidence finished")
            return connected_var_set, connected_edge_set, connected_feature_set
        else:
            raise ValueError('input type error')