#提供挑选证据的一些方法
import logging
import random


class EvidenceSelect:
    def __init__(self, variables, features, interval_evidence_count = 200, subgraph_max_num=3000,each_feature_evidence_limit = 2000):
        self.variables = variables  #变量集合
        self.features = features    #特征集合
        self.subgraph_max_num = subgraph_max_num    #子图允许的最大变量个数
        self.interval_evidence_count = interval_evidence_count   #，统一划分成10个区间，每个区间采样的证据变量个数
        self.each_feature_evidence_limit = each_feature_evidence_limit   #限制子图中每个单因子的证据变量数目

    def evidence_select(self, var_id):
        '''
        统一的证据选择方法
        connected_var_set = set()  子图变量集合
        connected_edge_set = set() 子图边集合 单因子[feature_id,var_id],双因子[feature_id,(id1,id2)]
        connected_feature_set = set() 子图特征(因子)集合
        '''
        if type(var_id) == int:
            subgraph_max_num = self.subgraph_max_num
            random_sample_num = self.each_feature_evidence_limit  # 没有feratureValue时每个单因子要采样的证据数目
            connected_var_set = set()  # 最后再添加隐变量id
            connected_edge_set = set()  # [feature_id,var_id]
            connected_feature_set = set()
            feature_set = self.variables[var_id]['feature_set']
            binary_feature_set = set()
            unary_feature_set = set()
            # potential_var_set = set()
            current_var_set = set() #当前hop的基础变量
            current_var_set.add(var_id)
            next_var_set = set()
            k_hop = 2
            # 划分此隐变量的双因子和单因子
            for feature_id in feature_set.keys():
                if self.features[feature_id]['feature_type'] == 'binary_feature':
                    binary_feature_set.add(feature_id)
                elif self.features[feature_id]['feature_type'] == 'unary_feature':
                    unary_feature_set.add(feature_id)
            #处理双因子，找k-hop跳的证据变量
            for k in range(k_hop):
                # 每轮添加上一轮隐变量的邻接证据变量
                for varid in current_var_set:
                    feature_set = self.variables[varid]['feature_set']
                    for feature_id in feature_set.keys():
                        # relation型特征为双因子
                        # 若此隐变量id包含在某个双因子相连的两个变量id中，且另一变量是证据变量，则将此证据变量加入下一轮next_var_set中，并且统计相关的特征和边
                        if self.features[feature_id]['feature_type'] == 'binary_feature':
                            weight = self.features[feature_id]['weight']
                            for id in weight.keys():
                                if type(id) == tuple and varid in id:
                                    another_var_id = id[0] if id[0] != varid else id[1]
                                    if self.variables[another_var_id]['is_evidence'] == True:
                                        next_var_set.add(another_var_id)
                                        connected_feature_set.add(feature_id)
                                        connected_edge_set.add((feature_id, id))   #[feature_id,(id1,id2)]
                    connected_var_set = connected_var_set.union(next_var_set)
                    current_var_set = next_var_set
                    next_var_set.clear()
            # 处理单因子：限制每个单因子的证据变量数目，需要时进行采样
            subgraph_capacity = subgraph_max_num - len(connected_var_set) - 1  # 最多可再添加的变量数目
            unary_evidence_set = set()
            unary_potential_set = set()
            # 先判断所有单因子证据加进来有没有超出最大限制
            for feature_id in unary_feature_set:
                weight = self.features[feature_id]['weight']
                for vid in weight.keys():
                    if self.variables[vid]['is_evidence'] == True:
                        unary_evidence_set.add(vid)
                    else:
                        unary_potential_set.add(vid)
            # 如果没有超出子图最大容量则全部加入
            if len(unary_evidence_set) <= subgraph_capacity:
                for feature_id in unary_feature_set:
                    weight = self.features[feature_id]['weight']
                    for vid in weight.keys():
                        if self.variables[vid]['is_evidence'] == True:
                            connected_var_set.add(vid)
                            connected_feature_set.add(feature_id)
                            connected_edge_set.add((feature_id, vid))  # [feature_id,id1]
            # 如果超出，则限制每个单因子的证据数目，按照是否有feature_value进行采样
            if len(unary_evidence_set) > subgraph_capacity:
                for feature_id in unary_feature_set:
                    # 有feature_value的按照区间采样
                    if self.features[feature_id]['parameterize'] == 1:
                        if self.features[feature_id]['evidence_count'] > 0 and self.features[feature_id]['monotonicity'] == True:
                            connected_feature_set.add(feature_id)
                            evidence_interval = self.features[feature_id]['evidence_interval']
                            for interval in evidence_interval:
                                # 如果这个区间的证据变量小于200，就全加进来
                                if len(interval) <= self.interval_evidence_count:
                                    connected_var_set = connected_var_set.union(interval)
                                    for vid in interval:
                                        connected_edge_set.add((feature_id, vid))
                                else:
                                    # 如果大于200,就随机采样200个
                                    sample = random.sample(list(interval), self.interval_evidence_count)
                                    connected_var_set = connected_var_set.union(sample)
                                    for vid in sample:
                                        connected_edge_set.add((feature_id, vid))     #(feature_id,v_id)
                    # 没有feature_value的随机采样
                    if self.features[feature_id]['parameterize'] == 0 and self.features[feature_id]['evidence_count'] > 0:
                        connected_feature_set.add(feature_id)
                        weight = self.features[feature_id]['weight']
                        unary_feature_evidence = set()  # 此特征上连接的所有证据变量的集合
                        for vid in weight.keys():
                            if self.variables[vid]['is_evidence'] == True:
                                unary_feature_evidence.add(vid)
                        sample = random.sample(list(unary_feature_evidence), random_sample_num)
                        connected_var_set = connected_var_set.union(sample)
                        unary_feature_evidence.clear()
                        for vid in sample:
                            connected_edge_set.add((feature_id, vid))
            #如果子图太小，只考虑没有参数化，则添加单因子上的隐变量，
            subgraph_capacity = subgraph_max_num - len(connected_var_set) - 1
            unary_connected_unlabeled_var = list()
            unary_connected_unlabeled_edge = list()
            unary_connected_unlabeled_feature = list()
            feature_set = self.variables[var_id]['feature_set']
            for feature_id in feature_set.keys():
                if self.features[feature_id]['feature_type'] == 'unary_feature' and self.features[feature_id]['parameterize'] == 0:
                    weight = self.features[feature_id]['weight']
                    for id in weight.keys():
                        # 先求出证据变量和隐变量的集合，包括相关的边和特征
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
            # 添加目标隐变量相关结构
            connected_var_set.add(var_id)
            for feature_id in unary_feature_set:
                if self.features[feature_id]['evidence_count'] > 0:
                    connected_edge_set.add((feature_id, var_id))
            logging.info("select evidence finished")
            return connected_var_set, connected_edge_set, connected_feature_set
        else:
            raise ValueError('input type error')

    def select_evidence_by_interval(self, var_id):
        '''
        按照feature_value的区间为指定的隐变量挑一定数量的证据变量,适用于ER
        主要思路：找到此隐变量的所有feature,针对每一个feature,将featureValue划分区间，每个区间选取200个变量
        @param var_id:
        @return:
        '''
        connected_var_set = set()
        connected_edge_set = set()
        connected_feature_set = set()  # 记录此隐变量上建因子图时实际保留了哪些feature
        feature_set = self.variables[var_id]['feature_set']
        #先加入证据变量和特征的边
        for feature_id in feature_set.keys():
            if self.features[feature_id]['evidence_count'] > 0:  # 有些feature上没有连接证据变量，就不用再加进来
                if self.features[feature_id]['monotonicity']==True and self.features[feature_id]['alpha_bound'][0] < self.features[feature_id]['alpha_bound'][1]:   #检查是否满足单调性假设
                    connected_feature_set.add(feature_id)
                    evidence_interval = self.features[feature_id]['evidence_interval']
                    for interval in evidence_interval:
                        # 如果这个区间的证据变量小于200，就全加进来
                        if len(interval) <= self.interval_evidence_count:
                            connected_var_set = connected_var_set.union(interval)
                            for id in interval:
                                connected_edge_set.add((feature_id, id))
                        else:
                            # 如果大于200,就随机采样200个
                            sample = random.sample(list(interval), self.interval_evidence_count)
                            connected_var_set = connected_var_set.union(sample)
                            for id in sample:
                                connected_edge_set.add((feature_id, id))

        #为了保证隐变量放在最后，，此处需将set转为list
        connected_var_set = list(connected_var_set)
        connected_edge_set = list(connected_edge_set)
        # 再加入此隐变量和特征的边
        connected_var_set.append(var_id)
        for feature_id in connected_feature_set:
            connected_edge_set.append((feature_id,var_id))
        logging.info("var-" + str(var_id) + " select evidence by interval finished")
        return connected_var_set, connected_edge_set, connected_feature_set

    def select_evidence_by_relation(self, var_id):
        '''
        为选出的top_k个隐变量挑选证据，适用于ALSA
        输入：
        var_id - -- 待构建子图的变量id
        subgraph_max_num - -子图允许的最大变量个数
        k_hop - - 找相邻变量的跳数

        输出：
        connected_var_set - -证据变量的id集合
        connected_edge_set - - 边的集合
        connected_feature_set - -能用得上的feature的集合
        '''
        if type(var_id) == int:
            subgraph_max_num = self.subgraph_max_num
            k_hop = 2
            connected_var_set = set()
            connected_edge_set = set()
            connected_feature_set = set()  # 记录此隐变量上建因子图时实际保留了哪些feature
            # connected_var_set = connected_var_set.union(set(var_id_list))
            connected_var_set.add(var_id)
            current_var_set = connected_var_set
            next_var_set = set()
            # 先找relation型特征的k-hop跳的证据变量(需确定此处是否只添加证据变量，不包括隐变量)
            for k in range(k_hop):
                # 每轮添加上一轮隐变量的邻接证据变量
                for varid in current_var_set:
                    feature_set = self.variables[varid]['feature_set']
                    for feature_id in feature_set.keys():
                        # relation型特征为双因子
                        # 若此隐变量id包含在某个双因子相连的两个变量id中，且另一变量是证据变量，则将此证据变量加入下一轮next_var_set中，并且统计相关的特征和边
                        if self.features[feature_id]['feature_type'] == 'binary_feature':
                            weight = self.features[feature_id]['weight']
                            for id in weight.keys():
                                if type(id) == tuple and varid in id:
                                    another_var_id = id[0] if id[0] != varid else id[1]
                                    if self.variables[another_var_id]['is_evidence'] == True:
                                        next_var_set.add(another_var_id)
                                        connected_feature_set.add(feature_id)
                                        connected_edge_set.add((feature_id, id))
                    connected_var_set = connected_var_set.union(next_var_set)
                    current_var_set = next_var_set
                    next_var_set.clear()
            # 再找和这k个变量共享word型feature的变量（先加证据变量，如果没有超过最大变量限制，再加隐变量）
            subgraph_capacity = subgraph_max_num - len(connected_var_set)  # 最多可再添加的变量数目
            unary_connected_unlabeled_var = list()
            unary_connected_unlabeled_edge = list()
            unary_connected_unlabeled_feature = list()
            unary_connected_evidence_var = list()
            unary_connected_evidence_edge = list()
            unary_connected_evidence_feature = list()
            # for var_id in var_id_list:
            feature_set = self.variables[var_id]['feature_set']
            for feature_id in feature_set.keys():
                if self.features[feature_id]['feature_type'] == 'unary_feature':
                    weight = self.features[feature_id]['weight']
                    for id in weight.keys():
                        # 先求出证据变量和隐变量的集合，包括相关的边和特征
                        if self.variables[id]['is_evidence'] == True:
                            unary_connected_evidence_var.append(id)
                            unary_connected_evidence_feature.append(feature_id)
                            unary_connected_evidence_edge.append((feature_id, id))
                        else:
                            unary_connected_unlabeled_var.append(id)
                            unary_connected_unlabeled_feature.append(feature_id)
                            unary_connected_unlabeled_edge.append((feature_id, id))
            # 限制子图规模大小
            if (len(unary_connected_evidence_var) <= subgraph_capacity):
                # 若证据变量个数小于当前可容变量个数，添加所有证据变量，再添加隐变量
                connected_var_set = connected_var_set.union(set(unary_connected_evidence_var))
                connected_feature_set = connected_feature_set.union((set(unary_connected_evidence_feature)))
                connected_edge_set = connected_edge_set.union(set(unary_connected_evidence_edge))
                # 若隐变量个数小于当前可容变量个数，添加所有隐变量
                if (len(unary_connected_unlabeled_var) <= (subgraph_capacity - len(unary_connected_evidence_var))):
                    connected_var_set = connected_var_set.union(set(unary_connected_unlabeled_var))
                    connected_feature_set = connected_feature_set.union((set(unary_connected_unlabeled_feature)))
                    connected_edge_set = connected_edge_set.union(set(unary_connected_unlabeled_edge))
                # 若隐变量个数大于当前可容变量个数，添加最多允许的隐变量个数
                else:
                    connected_var_set = connected_var_set.union(
                        set(unary_connected_unlabeled_var[:subgraph_capacity - len(unary_connected_evidence_var)]))
                    connected_feature_set = connected_feature_set.union(
                        set(unary_connected_unlabeled_feature[:subgraph_capacity - len(unary_connected_evidence_var)]))
                    connected_edge_set = connected_edge_set.union(
                        set(unary_connected_unlabeled_edge[:subgraph_capacity - len(unary_connected_evidence_var)]))
            else:
                # 若证据变量个数大于当前可容变量个数，添加最多允许的证据变量个数
                connected_var_set = connected_var_set.union(set(unary_connected_evidence_var[:subgraph_capacity]))
                connected_feature_set = connected_feature_set.union(
                    (set(unary_connected_evidence_feature[:subgraph_capacity])))
                connected_edge_set = connected_edge_set.union(set(unary_connected_evidence_edge[:subgraph_capacity]))
            logging.info("select evidence by relation finished")
            return connected_var_set, connected_edge_set, connected_feature_set
        else:
            raise ValueError('input type error')
