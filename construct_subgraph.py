import random
from collections import namedtuple
import logging
from numbskull_extend.numbskulltypes import *

    
class ConstructSubgraph:

    def __init__(self, variables, features,balance):
        self.variables = variables  #因子图变量
        self.features = features    #因子图特征
        self.balance = balance      #是否需要平衡证据变量中0和1的数目

    def construct_subgraph(self, evidences,var_id):
        connected_var_set, connected_edge_set, connected_feature_set = evidences
        var_map = dict()  # 用来记录self.variables与numbskull的variable变量id的映射-(self,numbskull)
        # 1.初始化variable
        var_num = len(connected_var_set)
        variable = np.zeros(var_num, Variable)
        variable_index = 0
        for id in connected_var_set:
            variable[variable_index]["isEvidence"] = self.variables[id]['is_evidence']  # 是证据变量为True，否则为False
            variable[variable_index]["initialValue"] = self.variables[id]['label']  # 变量的初始值
            variable[variable_index]["dataType"] = 0  # datatype=0表示是bool变量，为1时表示非布尔变量。
            variable[variable_index]["cardinality"] = 2  #二分类
            var_map[id] = variable_index  # 一一记录映射关系
            variable_index += 1
        #2.初始化weight: numbskull中的weight对应外部的feature,多个因子可以共享同一个weight
        weight = np.zeros(len(connected_feature_set), Weight)  # weight的数目等于feature的数目
        alpha_bound = np.zeros(len(connected_feature_set), AlphaBound)
        tau_bound = np.zeros(len(connected_feature_set), TauBound)
        feature_map_weight = dict()  # 记录feature id和weight id之间的映射 [feature_id,weight_id]
        weight_map_feature = dict()  # 用来记录weight id与feature id的映射 [weight_id,feature_id]
        weight_map_factor = dict()  # 用来记录每一个weight拥有的factor
        weight_index = 0
        for feature_id in connected_feature_set:
            weight[weight_index]["isFixed"] = False  # 因子图学习时，权重值固定为True，否则为False
            weight[weight_index]["parameterize"] = self.features[feature_id]['parameterize']  #权重是是否需要参数化
            #如果需要参数化，则需要初始化参数值及上下界上下界
            if self.features[feature_id]['parameterize'] == 1:
                weight[weight_index]["a"] = self.features[feature_id]['tau']
                weight[weight_index]["b"] = self.features[feature_id]['alpha']
                alpha_bound[weight_index]['lowerBound'] = self.features[feature_id]['alpha_bound'][0]
                alpha_bound[weight_index]['upperBound'] = self.features[feature_id]['alpha_bound'][1]
                tau_bound[weight_index]['lowerBound'] = self.features[feature_id]['tau_bound'][0]
                tau_bound[weight_index]['upperBound'] = self.features[feature_id]['tau_bound'][1]
            key = (random.sample(self.features[feature_id]['weight'].keys(), 1))[0]
            weight[weight_index]["initialValue"] = self.features[feature_id]['weight'][key][0]  # 此处一个weight可能会有很多个weight_value,赋值为第一个
            feature_map_weight[feature_id] = weight_index  # 一一记录映射关系
            weight_map_feature[weight_index] = feature_id
            weight_map_factor[weight_index] = set()
            weight_index += 1
        #3.初始化factor,fmap,edges
        binary_feature_edge = list()  # 双因子边的集合
        unary_feature_edge = list()   #单因子边的集合
        for elem in connected_edge_set:   #elem: [feature_id,(varid1,varid2)] or  [feature_id,varid]
            if self.features[elem[0]]['feature_type'] == 'unary_feature':
                unary_feature_edge.append(elem)
            elif self.features[elem[0]]['feature_type'] == 'binary_feature':
                binary_feature_edge.append(elem)
        edges_num = len(unary_feature_edge) + 2 * len(binary_feature_edge)  # 边的数目=单因子数目+2*双因子数目
        factor = np.zeros(len(unary_feature_edge) + len(binary_feature_edge), Factor)  # 因子的数目=单因子数目+双因子数目
        fmap = np.zeros(edges_num, FactorToVar)  # 记录边的映射关系factor[factor_index]->fmp_index，fmap[fmp_index]->var_index
        domain_mask = np.zeros(var_num, np.bool)
        edges = list()
        unary_edge = namedtuple('unary_edge', ['index', 'factorId', 'varId'])  # 单变量因子的边
        binary_edge = namedtuple('binary_edge', ['index', 'factorId', 'varId1', 'varId2'])  # 双变量因子的边
        factor_index = 0
        fmp_index = 0
        edge_index = 0
        # 初始化单因子，单因子暂用因子函数13
        for elem in unary_feature_edge:  # [feature_id,var_id]
            feature_id = elem[0]
            vid = elem[1]  #边上的变量id,后面用来查找feature_value
            factor[factor_index]["factorFunction"] = 13
            factor[factor_index]["weightId"] = feature_map_weight[feature_id]  #因子对应的权重id
            factor[factor_index]["featureValue"] = 1  # 因子特征值
            factor[factor_index]["arity"] = 1  # 单因子度为1
            factor[factor_index]["ftv_offset"] = fmp_index  # 偏移量每次加1
            fmap[fmp_index]["vid"] = var_map[vid]
            if self.features[feature_id]['parameterize'] == 1:
                fmap[fmp_index]["x"] = self.variables[vid]['feature_set'][feature_id][1]  # feature_value
                fmap[fmp_index]["theta"] = self.variables[vid]['feature_set'][feature_id][0]  # theta
            weight_map_factor[feature_map_weight[elem[0]]].add(factor_index) #记录此因子从属于哪个weight
            edges.append(unary_edge(edge_index, factor_index, var_map[vid]))
            fmp_index += 1
            factor_index += 1
            edge_index += 1
        # 初始化双因子，双因子暂用因子函数9
        for elem in binary_feature_edge:  # [feature_id,（var_id1，var_id2)]
            feature_id = elem[0]
            vid = elem[1]
            factor[factor_index]["factorFunction"] = 9  # 选取9号因子函数
            factor[factor_index]["weightId"] = feature_map_weight[feature_id]  # 因子对应的权重id
            factor[factor_index]["featureValue"] = 1  #因子特征值
            factor[factor_index]["arity"] = 2  # 双因子度为2
            factor[factor_index]["ftv_offset"] = fmp_index  #偏移量每次加2
            weight_map_factor[feature_map_weight[feature_id]].add(factor_index)
            edges.append(binary_edge(edge_index, factor_index, var_map[vid[0]], var_map[vid[1]]))
            for id in vid:
                fmap[fmp_index]["vid"] = var_map[id]
                # 支持后续的双因子函数化
                if self.features[feature_id]['parameterize'] == 1:
                    fmap[fmp_index]["x"] = self.variables[vid]['feature_set'][feature_id][1]  # feature_value
                    fmap[fmp_index]["theta"] = self.variables[vid]['feature_set'][feature_id][0]  # theta
                fmp_index += 1
            factor_index += 1
            edge_index += 1
        # 平衡化：扩充较少一边的变量数目
        if self.balance:
            #生成sampleList用于平衡化
            label0_var = list()
            label1_var = list()
            poential_var = list()  #子图中不止存在一个隐变量
            for id in range(0,var_num):
                if variable[id]['initialValue'] == 0:
                    label0_var.append(id)
                elif variable[id]['initialValue'] == 1:
                    label1_var.append(id)
                elif id != var_id:
                    poential_var.append(id)
            if len(label0_var)>=len(label1_var):
                extend_1 = True
                diff = len(label0_var)-len(label1_var)
            else:
                diff = len(label1_var) - len(label0_var)
                extend_1 = False
            #sample_num = len(label1_var) + len(label0_var) + diff + 1+len(poential_var)
            sample_num = len(connected_var_set)+diff
            index_list = [x for x in range(0,sample_num-1)]
            random.shuffle(index_list)  #为了打乱0-1顺序
            sample_list = np.zeros(sample_num, SampleList)
            sample_index = 0
            #添加所有0
            for id in label0_var:
                sample_list[index_list[sample_index]]['vid'] = id
                sample_index += 1
            #添加所有1
            for id in label1_var:
                sample_list[index_list[sample_index]]['vid'] = id
                sample_index += 1
            #扩充差异的部分
            if diff>0:
                if extend_1:
                   for i in range(0,diff):
                       sample_list[index_list[sample_index]]['vid'] = random.choice(label1_var)
                       sample_index += 1
                elif not extend_1:
                    for i in range(0, diff):
                        sample_list[index_list[sample_index]]['vid'] = random.choice(label0_var)
                        sample_index += 1
            #添加原来的隐变量（除了目标隐变量）
            for id in enumerate(poential_var):
                sample_list[index_list[sample_index]]['vid'] = id
                sample_index += 1
            #目标隐变量添加在最后
            sample_list[sample_num-1]['vid'] = var_id

        # 初始化wmap(WeightToFactor))和wfactor(FactorToWeight),用于批量梯度下降
        wmap = np.zeros(len(weight), WeightToFactor)  #用于查找每一个weight关联的所有factor
        wfactor = np.zeros(len(factor), FactorToWeight)  # 按weight顺序组织的factor
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
        logging.info("construct subgraph for mixture succeed")
        return weight, variable, factor, fmap, domain_mask, edges_num, var_map, alpha_bound, tau_bound, weight_map_feature, sample_list, wmap, wfactor

    def construct_subgraph_for_mixture(self,evidences,var_id):
        '''
        建立既有单因子，又有双因子的子图，适用于alsa
        :param evidences: 构建因子图时所需的变量，边，特征集合
        :return: Numbskull推理所需数据（weight, variable, factor, fmap, domain_mask, edges_num, var_map）
        '''
        connected_var_set, connected_edge_set, connected_feature_set = evidences
        var_map = dict()   #用来记录self.variables与numbskull的variable变量的映射-(self,numbskull)
        #初始化变量
        var_num = len(connected_var_set)
        variable = np.zeros(var_num, Variable)
        variable_index = 0
        for id in connected_var_set:
            variable[variable_index]["isEvidence"] = self.variables[id]['is_evidence']  #是证据变量为True，否则为False
            variable[variable_index]["initialValue"] = self.variables[id]['label']  #变量的初始值
            variable[variable_index]["dataType"] = 0    # datatype=0表示是bool变量，为1时表示非布尔变量。
            variable[variable_index]["cardinality"] = 2
            var_map[id] = variable_index     #一一记录映射关系
            variable_index += 1
        #初始化weight,多个因子可以共享同一个weight
        weight = np.zeros(len(connected_feature_set), Weight)  #weight的数目等于feature的数目
        wmap = np.zeros(len(weight), WeightToFactor)  # 用于查找每一个weight关联的所有factor
        feature_map_weight = dict()  # 记录feature id和weight id之间的映射 [feature_id,weight_id]
        weight_map = dict()  # 用来记录weight id与feature id的映射
        weight_dict = dict()  # 用来记录每一个weight拥有的factor
        weight_index = 0
        for feature_id in connected_feature_set:
            weight[weight_index]["isFixed"] = False  #因子图学习时，权重值固定为True，否则为False
            weight[weight_index]["parameterize"] = False    #权重需要参数化，则为True，否则为False
            key = (random.sample(self.features[feature_id]['weight'].keys(),1))[0]
            weight[weight_index]["initialValue"] = self.features[feature_id]['weight'][key][0] #此处一个weight可能会有很多个weight_value,赋值为第一个
            feature_map_weight[feature_id] = weight_index   #一一记录映射关系
            weight_map[weight_index] = feature_id
            weight_dict[weight_index] = set()
            weight_index += 1

        #先划分单因子双因子
        binary_feature_edge = list()    #双因子边的集合
        unary_feature_edge = list()     #单因子边的集合
        for elem in connected_edge_set:
            if self.features[elem[0]]['feature_type'] == 'unary_feature':
                unary_feature_edge.append(elem)
            elif self.features[elem[0]]['feature_type'] == 'binary_feature':
                binary_feature_edge.append(elem)
        #初始化factor,fmap,edges
        edges_num = len(unary_feature_edge) + 2*len(binary_feature_edge) # 边的数目=单因子数目+2*双因子数目
        factor = np.zeros(len(unary_feature_edge)+len(binary_feature_edge), Factor)  #因子的数目=单因子数目+双因子数目
        wfactor = np.zeros(len(factor), FactorToWeight)  # 按weight顺序组织的factor
        fmap = np.zeros(edges_num, FactorToVar)  #记录边的映射关系factor[factor_index]->fmp_index，fmap[fmp_index]->var_index
        domain_mask = np.zeros(var_num, np.bool)
        edges = list()
        unary_edge = namedtuple('unary_edge', ['index', 'factorId', 'varId'])  # 单变量因子的边
        binary_edge = namedtuple('binary_edge', ['index', 'factorId', 'varId1', 'varId2'])  # 双变量因子的边
        edge = namedtuple('edge', ['index', 'factorId', 'varId'])  # 单变量因子的边
        factor_index = 0
        fmp_index = 0
        edge_index = 0
        #初始化单变量因子
        for elem in unary_feature_edge:  # [feature_id,var_id]
            var_index = elem[1]
            factor[factor_index]["factorFunction"] = 18 #选取18号因子函数
            factor[factor_index]["weightId"] = feature_map_weight[elem[0]]  #因子对应的权重id
            factor[factor_index]["featureValue"] = 1    #因子特征值
            factor[factor_index]["arity"] = 1  # 单因子度为1
            factor[factor_index]["ftv_offset"] = fmp_index  # 偏移量每次加1
            weight_dict[feature_map_weight[elem[0]]].add(factor_index)
            edges.append(unary_edge(edge_index, factor_index, var_map[elem[1]]))
            fmap[fmp_index]["vid"] = var_map[elem[1]]
            fmp_index += 1
            factor_index += 1
            edge_index += 1
        #初始化双变量因子
        for elem in binary_feature_edge:  # [feature_id,（var_id1，var_id2)]
            var_index = elem[1]
            factor[factor_index]["factorFunction"] = 9  #选取9号因子函数
            factor[factor_index]["weightId"] = feature_map_weight[elem[0]]  #因子对应的权重id
            factor[factor_index]["featureValue"] = 1    #因子特征值
            factor[factor_index]["arity"] = 2  # 双因子度为2
            factor[factor_index]["ftv_offset"] = fmp_index  # 偏移量每次加1
            weight_dict[feature_map_weight[elem[0]]].add(factor_index)
            edges.append(binary_edge(edge_index, factor_index, var_map[elem[1][0]],var_map[elem[1][1]]))
            fmap[fmp_index]["vid"] = var_map[elem[1][0]]
            fmp_index += 1
            fmap[fmp_index]["vid"] = var_map[elem[1][1]]
            fmp_index += 1
            factor_index += 1
            edge_index += 1
        logging.info("construct subgraph for mixture succeed")

        alpha_bound = np.zeros(len(connected_feature_set), AlphaBound)
        tau_bound = np.zeros(len(connected_feature_set), TauBound)
        sample_list = None
        # 初始化wmap(WeightToFactor))和wfactor(FactorToWeight),用于批量梯度下降
        wfactor_index = 0
        for weightId, factorSet in weight_dict.items():
            count = 0
            wmap[weightId]["weightId"] = weightId
            wmap[weightId]["weight_index_offset"] = wfactor_index
            for factorId in factorSet:
                wfactor[wfactor_index]["factorId"] = factorId
                count += 1
                wfactor_index += 1
            wmap[weightId]["weight_index_length"] = count
        return weight, variable, factor, fmap, domain_mask, edges_num, var_map,alpha_bound, tau_bound, weight_map, sample_list, wmap, wfactor


    def construct_subgraph_for_unaryPara(self,evidences,var_id):
        '''
           # 建立只有单因子，并且所有单因子都需要参数化的子图，适用于ER
           factor个数等于edge的个数，因此factor个数一定大于等于variable
        :param evidences:
        :return:
        '''
        connected_var_set, connected_edge_set, connected_feature_set = evidences   #(feature_id,var_id)
        # 初始化variable
        var_map = dict()  # 用来记录self.variables与numbskull的variable变量的映射-(self,numbskull)
        var_num = len(connected_var_set)    # 变量个数=证据变量+隐变量
        variable = np.zeros(var_num, Variable)
        variable_index= 0
        for id in connected_var_set:
            variable[variable_index]["isEvidence"] = self.variables[id]['is_evidence']  #是证据变量为True，否则为False
            variable[variable_index]["initialValue"] = self.variables[id]['label']      #变量的初始值
            variable[variable_index]["dataType"] = 0  # datatype=0表示是bool变量，为1时表示非布尔变量。
            variable[variable_index]["cardinality"] = 2
            var_map[id] = variable_index    # 一一记录映射关系
            variable_index += 1
        # 初始化weight,多个因子可以共享同一个weight
        weight = np.zeros(len(connected_feature_set), Weight)  # weight的数目等于feature的数目
        wmap = np.zeros(len(weight), WeightToFactor)  #用于查找每一个weight关联的所有factor
        alpha_bound = np.zeros(len(connected_feature_set), AlphaBound)
        tau_bound = np.zeros(len(connected_feature_set), TauBound)
        feature_map_weight = dict()  # 需要记录feature id和weight id之间的映射 [feature_id,weight_id]
        weight_map = dict()  # 用来记录weight id与feature id的映射
        weight_dict = dict()   #用来记录每一个weight拥有的factor
        weight_index = 0
        for feature_id in connected_feature_set:
            alpha_bound[weight_index]['lowerBound'] = self.features[feature_id]['alpha_bound'][0]
            alpha_bound[weight_index]['upperBound'] = self.features[feature_id]['alpha_bound'][1]
            tau_bound[weight_index]['lowerBound'] = self.features[feature_id]['tau_bound'][0]
            tau_bound[weight_index]['upperBound'] = self.features[feature_id]['tau_bound'][1]
            weight[weight_index]["isFixed"] = False     #因子图学习时，权重值固定为True，否则为False
            weight[weight_index]["parameterize"] = True     #权重需要参数化，则为True，否则为False
            #实体识别任务中权重的两个参数tau和alpha
            weight[weight_index]["a"] = self.features[feature_id]['tau']
            weight[weight_index]["b"] = self.features[feature_id]['alpha']
            weight[weight_index]["initialValue"] = random.uniform(-5,5)  # 此处一个weight会有很多个weight_value，此处随机初始化一个，后面应该用不上
            feature_map_weight[feature_id] = weight_index   #一一记录映射关系
            weight_map[weight_index] = feature_id
            weight_dict[weight_index] = set()
            weight_index += 1

        #初始化factor,fmap,edges
        edges_num = len(connected_edge_set)         # 边的数目
        factor = np.zeros(edges_num, Factor)        # 实体识别任务中只有单因子，所以有多少个边就有多少个因子
        fmap = np.zeros(edges_num, FactorToVar)     #记录边的映射关系factor[factor_index]->fmp_index，fmap[fmp_index]->var_index
        wfactor= np.zeros(len(factor), FactorToWeight)   #按weight顺序组织的factor
        domain_mask = np.zeros(var_num, np.bool)
        edges = list()
        edge = namedtuple('edge', ['index', 'factorId', 'varId'])  #单变量因子的边
        factor_index = 0
        fmp_index = 0
        edge_index = 0
        for elem in connected_edge_set:  # [feature_id,var_id]
            var_index = elem[1]
            factor[factor_index]["factorFunction"] = 13     #选取9号因子函数
            factor[factor_index]["weightId"] = feature_map_weight[elem[0]]  #因子对应的权重id
            weight_dict[feature_map_weight[elem[0]]].add(factor_index)
            factor[factor_index]["featureValue"] =1 #self.variables[var_index]['feature_set'][elem[0]][1] #因子特征值，在实体识别任务中为特征和变量的相似度
            factor[factor_index]["arity"] = 1  # 单因子度为1
            factor[factor_index]["ftv_offset"] = fmp_index  # 偏移量每次加1
            edges.append(edge(edge_index, factor_index, var_map[elem[1]]))
            fmap[fmp_index]["vid"] = edges[factor_index][2]
            fmap[fmp_index]["x"] = self.variables[var_index]['feature_set'][elem[0]][1]  # feature_value
            fmap[fmp_index]["theta"] = self.variables[var_index]['feature_set'][elem[0]][0]  # theta
            fmp_index += 1
            factor_index += 1
            edge_index += 1

        if self.balance:
            #生成sampleList用于平衡化
            label0_var = list()
            label1_var = list()
            for id in range(0,var_num):
                if variable[id]['initialValue'] == 0:
                    label0_var.append(id)
                elif variable[id]['initialValue'] == 1:
                    label1_var.append(id)
                else:
                    poential_var = id
            if len(label0_var)>=len(label1_var):
                extend_1 = True
                diff = len(label0_var)-len(label1_var)
            else:
                diff = len(label1_var) - len(label0_var)
                extend_1 = False
            sample_num = len(label1_var) + len(label0_var) + diff + 1
            index_list = [x for x in range(0,sample_num-1)]
            random.shuffle(index_list)
            sample_list = np.zeros(sample_num, SampleList)
            sample_index = 0
            #添加所有0
            for id in label0_var:
                sample_list[index_list[sample_index]]['vid'] = id
                sample_index += 1
            #添加所有1
            for id in label1_var:
                sample_list[index_list[sample_index]]['vid'] = id
                sample_index += 1
            #扩充差异的部分
            if diff>0:
                if extend_1:
                   for i in range(0,diff):
                       sample_list[index_list[sample_index]]['vid'] = random.choice(label1_var)
                       sample_index += 1
                elif not extend_1:
                    for i in range(0, diff):
                        sample_list[index_list[sample_index]]['vid'] = random.choice(label0_var)
                        sample_index += 1
            #添加隐变量
            sample_list[sample_num-1]['vid'] = poential_var
        logging.info("construct subgraph for unaryPara succeed")

        #初始化wmap(WeightToFactor))和wfactor(FactorToWeight),用于批量梯度下降
        wfactor_index = 0
        for weightId, factorSet in weight_dict.items():
            count = 0
            wmap[weightId]["weightId"] = weightId
            wmap[weightId]["weight_index_offset"] = wfactor_index
            for factorId in factorSet:
                wfactor[wfactor_index]["factorId"] = factorId
                count += 1
                wfactor_index += 1
            wmap[weightId]["weight_index_length"] = count
        return weight, variable, factor, fmap,domain_mask, edges_num, var_map,alpha_bound,tau_bound,weight_map,sample_list,wmap,wfactor

    def construct_subgraph_for_custom(self, evidences,var_id):
        '''
        用户自定义建子图的方法
        :param var_id:
        :param evidences:
        :return:
        '''
        pass