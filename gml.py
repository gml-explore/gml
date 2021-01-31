import heapq
import pickle
import time
from copy import copy

from numbskull_extend import numbskull
import logging
from sklearn import metrics
import gml_utils
from evidential_support import EvidentialSupport
from evidence_select import EvidenceSelect
from approximate_probability_estimation import ApproximateProbabilityEstimation
from construct_subgraph import ConstructSubgraph
from configparser import ConfigParser
from concurrent.futures import ProcessPoolExecutor
class GML:
    '''
    GML主类: Evidential Support -> Approximate Probability Estimation -> select topm -> select topk -> inference -> label
    '''
    def __init__(self, dataname,variables, features, evidential_support_method, approximate_probability_method,
                 evidence_select_method, construct_subgraph_method, learning_method,top_m=2000, top_k=10, update_proportion= -1,
                 balance=False,optimization = False,optimization_threshold = 1e-6,learning_epoches = 1000,inference_epoches = 1000,nprocess=1):
        '''
        GML主类参数初始化
        @param dataname:
        @param variables:
        @param features:
        @param evidential_support_method:
        @param approximate_probability_method:
        @param evidence_select_method:
        @param construct_subgraph_method:
        @param top_m:
        @param top_k:
        @param update_proportion:
        @param balance:
        @param optimization:
        @param optimization_threshold:
        '''
        #check data
        variables_keys= ['var_id','is_easy','is_evidence','true_label','label','feature_set']
        features_keys = ['feature_id','feature_type','feature_name','weight']
        evidential_support_methods = ['regression','relation']
        approximate_probability_methods = ['interval','relation']
        evidence_select_methods = ['interval','relation']
        construct_subgraph_methods= ['unaryPara','mixture']
        if evidential_support_method not in evidential_support_methods:
            raise ValueError('evidential_support_method has no this method: '+evidential_support_method)
        if approximate_probability_method not in approximate_probability_methods:
            raise ValueError('approximate_probability_method has no this method: '+approximate_probability_method)
        if evidence_select_method not in evidence_select_methods:
            raise ValueError('evidence_select_method has no this method: '+evidence_select_method)
        if construct_subgraph_method not in construct_subgraph_methods:
            raise ValueError('construct_subgraph_method has no this method: '+construct_subgraph_method)
        for variable in variables:
            for attribute in variables_keys:
                if attribute not in variable:
                    raise ValueError('variables has no key: '+attribute)
        for feature in features:
            for attribute in features_keys:
                if attribute not in feature:
                    raise ValueError('features has no key: '+attribute)
        self.dataname = dataname
        self.variables = variables
        self.features = features
        self.evidential_support_method = evidential_support_method  # 选择evidential support的方法
        self.evidence_select_method = evidence_select_method  # 选择select evidence的方法
        self.approximate_probability_method = approximate_probability_method  # 选择估计近似概率的方法
        self.construct_subgraph_method = construct_subgraph_method  # 选择构建因子图的方法
        self.learning_method = learning_method  #梯度下降使用的方法（sgd或者bgd）
        self.labeled_variables_set = set()  # 所有新标记变量集合
        self.top_m = top_m
        self.top_k = top_k
        self.optimization = optimization
        self.optimization_threshold = optimization_threshold
        self.update_proportion = update_proportion
        self.observed_variables_set, self.poential_variables_set = gml_utils.separate_variables(variables)
        self.support = EvidentialSupport(variables, features, evidential_support_method)
        self.select = EvidenceSelect(variables, features)
        self.approximate = ApproximateProbabilityEstimation(variables,features,approximate_probability_method)
        self.subgraph = ConstructSubgraph(variables, features, balance)
        self.learing_epoches = learning_epoches
        self.inference_epoches = inference_epoches
        self.nprocess = nprocess
        self.now = str(time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())))
        #log
        logging.basicConfig(
            level=logging.INFO,  # 设置输出信息等级
            format='%(asctime)s - %(name)s - [%(levelname)s]: %(message)s'  # 设置输出格式
        )
        #save results
        with open("./results/"+self.dataname+'-'+self.now+'-result.txt', 'w') as f:
            f.write('var_id'+' '+'inferenced_probability'+' '+'inferenced_label'+' '+'ture_label'+'\n')

    @staticmethod
    def initial(configFile,variables,features):
        '''
        读取配置文件中的参数
        @param configFile:
        @param variables:
        @param features:
        @return:
        '''
        config = ConfigParser()
        config.read(configFile, encoding='UTF-8')
        dataname = config['para']['dataname']
        evidential_support_method = config['para']['evidential_support_method']
        approximate_probability_method = config['para']['approximate_probability_method']
        evidence_select_method = config['para']['evidence_select_method']
        construct_subgraph_method = config['para']['construct_subgraph_method']
        learning_method = config['para']['learning_method']
        learning_epoches = int(config['para']['learning_epoches'])
        inference_epoches = int(config['para']['inference_epoches'])
        top_m = int(config['para']['top_m'])
        top_k = int(config['para']['top_k'])
        update_proportion = float(config['para']['update_proportion'])
        balance = config['para'].getboolean('balance')
        optimization = config['para'].getboolean('optimization')
        optimization_threshold = float(config['para']['optimization_threshold'])
        return GML(dataname,variables, features, evidential_support_method, approximate_probability_method,
                 evidence_select_method, construct_subgraph_method,learning_method, top_m, top_k, update_proportion,
                 balance,optimization,optimization_threshold,learning_epoches,inference_epoches)

    def update_bound(self,var_id):
        '''
        在每轮标记完成后对参数上下界进行更新
        @param var_id:
        @return:
        '''
        feature_set = self.variables[var_id]['feature_set']
        for feature_id in feature_set.keys():
            feature_evidence0_count = 0
            feature_evidence1_count = 0
            feature_evidence0_sum = 0
            feature_evidence1_sum = 0
            weight = self.features[feature_id]['weight']
            for vid in weight.keys():
                if self.variables[vid]['is_evidence'] == True:
                    if self.variables[vid]['label'] == 0:
                        feature_evidence0_count += 1
                        feature_evidence0_sum += weight[vid][1]
                    elif self.variables[vid]['label'] == 1:
                        feature_evidence1_count += 1
                        feature_evidence1_sum += weight[vid][1]
            if feature_evidence0_count != 0:
                bound0 = feature_evidence0_sum / feature_evidence0_count
            else:
                bound0 = 0
            if feature_evidence1_count != 0:
                bound1 = feature_evidence1_sum / feature_evidence1_count
            else:
                bound1 = 0
            self.features[feature_id]['alpha_bound'] = copy([bound0, bound1])
            self.features[feature_id]['tau_bound'] = copy([-10, 10])


    def evidential_support(self, variable_set,update_feature_set):
        '''
        计算evidential support
        @param variable_set:
        @param update_feature_set:
        @return:
        '''
        method = 'self.support.evidential_support_by_' + self.evidential_support_method + '(variable_set,update_feature_set)'
        eval(method)

    def approximate_probability_estimation(self, variable_set):
        '''
        近似概率计算
        @param variable_set:
        @return:
        '''
        method = 'self.approximate.approximate_probability_estimation_by_' + self.approximate_probability_method + '(variable_set)'
        eval(method)

    def select_top_m_by_es(self, m):
        '''
        根据计算出的Evidential Support(从大到小)选前m个隐变量
        @param m:
        @return:
        '''
        #如果当前隐变量数目小于m,直接返回隐变量列表
        if m > len(self.poential_variables_set):
           return list(self.poential_variables_set)
        poential_var_list = list()
        m_id_list = list()
        for var_id in self.poential_variables_set:
            poential_var_list.append([var_id, self.variables[var_id]['evidential_support']])
        topm_var = heapq.nlargest(m, poential_var_list, key=lambda s: s[1])
        for elem in topm_var:
            m_id_list.append(elem[0])
        logging.info('select m finished')
        return m_id_list

    def select_top_k_by_entropy(self, var_id_list, k):
        '''
        计算熵，选出top_k个熵小的隐变量
        @param var_id_list:
        @param k:
        @return:
        '''
        #如果隐变量数目小于k,直接返回隐变量列表
        if len(var_id_list) < k:
            return var_id_list
        m_list = list()
        k_id_list = list()
        for var_id in var_id_list:
            m_list.append(self.variables[var_id])
        k_list = heapq.nsmallest(k, m_list, key=lambda x: x['entropy'])
        for var in k_list:
            k_id_list.append(var['var_id'])
        logging.info('select k finished')
        return k_id_list

    def select_evidence(self, var_id):
        '''
        为每个隐变量挑选证据
        @param var_id:
        @return:
        '''
        method = 'self.select.select_evidence_by_' + self.evidence_select_method + "(var_id)"
        connected_var_set, connected_edge_set, connected_feature_set = eval(method)
        return connected_var_set, connected_edge_set, connected_feature_set

    def construct_subgraph(self, var_id):
        '''
        构建子图
        @param var_id:
        @return:
        '''
        evidences = self.select_evidence(var_id)
        method = 'self.subgraph.construct_subgraph_for_' + self.construct_subgraph_method + "(evidences)"
        weight, variable, factor, fmap, domain_mask, edges_num, var_map,alpha_bound,tau_bound,weight_map,sample_list,wmap,wfactor = eval(method)
        return weight, variable, factor, fmap, domain_mask, edges_num, var_map,alpha_bound,tau_bound,weight_map,sample_list,wmap,wfactor

    def inference_subgraph(self, var_id):
        '''
        推理子图
        @param var_id: 待推理的变量id
        @return:
        '''
        if not (type(var_id) == set or type(var_id) == list  or type(var_id) == int):
            raise ValueError('var_id should be set,list,or int' )
        ns_learing = numbskull.NumbSkull(
            n_inference_epoch=self.learing_epoches,
            n_learning_epoch=self.inference_epoches,
            stepsize=0.01,
            decay=0.95,
            reg_param=1e-6,
            regularization=2,
            truncation=10,
            quiet=(not False),
            verbose=False,
            learn_non_evidence=True,  # need to test
            sample_evidence=False,
            burn_in=10,
            nthreads=1,
            learning_method = self.learning_method
        )
        weight, variable, factor, fmap, domain_mask, edges_num, var_map,alpha_bound,tau_bound,weight_map,sample_list,wmap,wfactor = self.construct_subgraph(var_id)
        subgraph = weight, variable, factor, fmap, domain_mask, edges_num,alpha_bound,tau_bound,sample_list,wmap,wfactor
        ns_learing.loadFactorGraph(*subgraph)
        # 因子图参数学习
        ns_learing.learning()
        logging.info("subgraph learning finished")
        # 对学习到的参数进行范围控制,并将weight的isfixed属性置为true
        for index, w in enumerate(weight):
            w["isFixed"] = True
            if self.evidence_select_method == 'interval':
                theta = self.variables[var_id]['feature_set'][weight_map[index]][0]
                x = self.variables[var_id]['feature_set'][weight_map[index]][1]
                a = ns_learing.factorGraphs[0].weight[index]['a']
                b = ns_learing.factorGraphs[0].weight[index]['b']
                w['initialValue'] = theta * a*(x-b)
            else:
                w['initialValue'] = ns_learing.factorGraphs[0].poential_weight[index]
        # 因子图推理
        ns_inference = numbskull.NumbSkull(
            n_inference_epoch=self.learing_epoches,
            n_learning_epoch= self.inference_epoches,
            stepsize=0.001,
            decay=0.95,
            reg_param=1e-6,
            regularization=2,
            truncation=10,
            quiet=(not False),
            verbose=False,
            learn_non_evidence=False,  # need to test
            sample_evidence=False,
            burn_in=10,
            nthreads=1,
            learning_method=self.learning_method
        )
        ns_inference.loadFactorGraph(*subgraph)
        ns_inference.inference()
        logging.info("subgraph inference finished")
        # 写回概率到self.variables
        if type(var_id) == set or type(var_id) == list:
            for id in var_id:
                self.variables[id]['inferenced_probability'] = ns_inference.factorGraphs[0].marginals[var_map[id]]
        elif type(var_id) == int:
            self.variables[var_id]['inferenced_probability'] = ns_inference.factorGraphs[0].marginals[var_map[var_id]]
        logging.info("inferenced probability recored")
        return var_id,ns_inference.factorGraphs[0].marginals[var_map[var_id]]

    def label(self, var_id_list):
        '''
        从topk个已推理变量中中选一个熵最小的进行标记
        @param var_id_list:
        @return:
        '''
        entropy_list = list()
        if len(var_id_list) > 1:  # 如果传入的变量个数大于1,就每次选熵最小的进行标记
            for var_id in var_id_list:
                var_index = var_id
                self.variables[var_index]['entropy'] = gml_utils.entropy(
                    self.variables[var_index]['inferenced_probability'])
                entropy_list.append([var_id, self.variables[var_index]['entropy']])
            min_var = heapq.nsmallest(1, entropy_list, key=lambda x: x[1])  # 选出熵最小的变量
            var = min_var[0][0]
        else:
            var = var_id_list[0]
        var_index = var  # 如果传入的只有1个变量，直接进行标记即可
        self.variables[var_index]['probability'] = self.variables[var_index]['inferenced_probability']
        self.variables[var_index]['label'] = 1 if self.variables[var_index]['probability'] >= 0.5 else 0
        self.variables[var_index]['is_evidence'] = True
        logging.info('var-' + str(var) + " labeled succeed---------------------------------------------")
        self.poential_variables_set.remove(var)
        self.observed_variables_set.add(var)
        self.labeled_variables_set.add(var)
        probability = self.variables[var_index]['probability']
        label = self.variables[var_index]['label']
        true_label = self.variables[var_index]['true_label']
        with open("./results/"+self.dataname+'-'+self.now+'-result.txt', 'a') as f:
            f.write(f'{var:7} {probability:10} {label:4} {true_label:4}')
            f.write('\n')
        return var

    def inference(self):
        '''主流程'''
        labeled_var = 0
        labeled_count = 0
        var = 0  #每轮标记的变量id
        m_list = list()
        update_feature_set = set()  # 存储一轮更新期间证据支持发生变化的feature
        inferenced_variables_id = set()  #一轮更新期间已经建立过因子图并推理的隐变量
        pool = ProcessPoolExecutor(self.nprocess)
        if self.update_proportion > 0:
            update_cache = int(self.update_proportion * len(self.poential_variables_set))  # 每推理update_cache个变量后需要重新计算evidential support
        self.evidential_support(self.poential_variables_set, None)
        self.approximate_probability_estimation(self.poential_variables_set)
        # 如果熵小于某个阈值，直接标记，不用推理
        if self.optimization == True:
            with open("./results/"+self.dataname+'-'+self.now+'-result.txt', 'a') as f:
                for vid in self.poential_variables_set:
                    if self.variables[vid]['entropy'] <= self.optimization_threshold:
                        self.variables[vid]['probability'] = self.variables[vid]['approximate_probability']
                        self.variables[vid]['is_evidence'] = True
                        self.variables[vid]['label'] = 1 if  self.variables[vid]['probability'] >= 0.5 else 0
                        if self.evidence_select_method == 'interval':
                            gml_utils.write_labeled_var_to_evidence_interval(self.variables, self.features, vid,self.support.evidence_interval)
                        logging.info('var-'+str(vid)+' labeled succeed---------------------------------------------')
                        probability = self.variables[vid]['probability']
                        label = self.variables[vid]['label']
                        true_label = self.variables[vid]['true_label']
                        f.write(f'{vid:7} {probability:10} {label:4} {true_label:4}')
                        f.write('\n')
                        self.labeled_variables_set.add(vid)
                f.write('Below are the variables for true inference>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'+'\n')
            self.observed_variables_set, self.poential_variables_set = gml_utils.separate_variables(self.variables)
            self.evidential_support(self.poential_variables_set, None)
            self.approximate_probability_estimation(self.poential_variables_set)
        while len(self.poential_variables_set) > 0:
            # update_proportion小于等于0表示不需要更新evidential support
            if self.update_proportion <= 0:
                self.evidential_support(self.poential_variables_set, None)
                self.approximate_probability_estimation(self.poential_variables_set)
            if  self.update_proportion > 0 and labeled_var == update_cache:
                #当标记的变量数目达到update_cache时，重新回归并计算evidential support
                for var_id in self.labeled_variables_set:
                    for feature_id in self.variables[var_id]['feature_set'].keys():
                        update_feature_set.add(feature_id)
                self.evidential_support(self.poential_variables_set, update_feature_set)
                self.approximate_probability_estimation(self.poential_variables_set)
                labeled_var = 0
                update_feature_set.clear()
                self.labeled_variables_set.clear()
                inferenced_variables_id.clear()
            m_list = self.select_top_m_by_es(self.top_m)
            k_list = self.select_top_k_by_entropy(m_list, self.top_k)
            '''
            if self.evidence_select_method == 'interval':
                #只要没有进行更新,就每次只推理新增的变量
                add_list = [x for x in k_list if x not in inferenced_variables_id]
                if len(add_list) > 0:
                    if(self.nprocess ==1): 
                        for var_id in add_list:
                            # if var_id not in inferenced_variables_id:
                            self.inference_subgraph(var_id)
                            # 每轮更新期间推理过的变量，因为参数没有更新，所以无需再进行推理。
                            inferenced_variables_id.add(var_id)
                    else:
                        futures = []
                        for var_id in add_list:
                            future = pool.submit(self.inference_subgraph,var_id)
                            futures.append(future)
                        #self.inference_subgraph(var_id)
                        # 每轮更新期间推理过的变量，因为参数没有更新，所以无需再进行推理。
                            inferenced_variables_id.add(var_id)
                        for ft in futures:
                            self.variables[ft.result()[0]]['inferenced_probability'] = ft.result()[1]
                var = self.label(k_list)
                gml_utils.write_labeled_var_to_evidence_interval(self.variables, self.features, var, self.support.evidence_interval)
                self.update_bound(var)   #每标记一个变量之后更新上下界
            else:
                self.inference_subgraph(k_list)
                var = self.label(k_list)
            '''
            add_list = [x for x in k_list if x not in inferenced_variables_id]
            if len(add_list) > 0:
                for var_id in add_list:
                    # if var_id not in inferenced_variables_id:
                    self.inference_subgraph(var_id)
                    # 每轮更新期间推理过的变量，因为参数没有更新，所以无需再进行推理。
                    inferenced_variables_id.add(var_id)
            var = self.label(k_list)
            if self.evidence_select_method == 'interval':
                gml_utils.write_labeled_var_to_evidence_interval(self.variables, self.features, var, self.support.evidence_interval)
                self.update_bound(var)  # 每标记一个变量之后更新上下界
            labeled_var += 1
            labeled_count += 1
            logging.info("label_count=" + str(labeled_count))
        #output results
        self.score()

    def save_model(self):
        '''
        推理完成后，保存经过推理后的因子图，应当保存变量，因子，学习后的权重等。
        @return:
        '''
        with open(self.dataname+'-'+self.now+"_variables.pkl",'wb') as v:
            pickle.dump(self.variables,v)
        with open(self.dataname+'-'+self.now+"_features.pkl",'wb') as v:
            pickle.dump(self.features,v)


    def score(self):
        '''
        计算推理结果的准确率，精确率，召回率，f1值等
        :return:
        '''
        easys_pred_label = list()
        easys_true_label = list()
        hards_pred_label = list()
        hards_true_label = list()
        for var in self.variables:
            if var['is_easy'] == True:
                easys_true_label.append(var['true_label'])
                easys_pred_label.append(var['label'])
            else:
                hards_true_label.append(var['true_label'])
                hards_pred_label.append(var['label'])

        all_true_label = easys_true_label + hards_true_label
        all_pred_label = easys_pred_label + hards_pred_label

        print("--------------------------------------------")
        print("total:")
        print("--------------------------------------------")
        print("total accuracy_score: " + str(metrics.accuracy_score(all_true_label, all_pred_label)))
        print("total precision_score: " + str(metrics.precision_score(all_true_label, all_pred_label)))
        print("total recall_score: " + str(metrics.recall_score(all_true_label, all_pred_label)))
        print("total f1_score: " + str(metrics.f1_score(all_true_label, all_pred_label)))
        print("--------------------------------------------")
        print("easys:")
        print("--------------------------------------------")
        print("easys accuracy_score:" + str(metrics.accuracy_score(easys_true_label, easys_pred_label)))
        print("easys precision_score:" + str(metrics.precision_score(easys_true_label, easys_pred_label)))
        print("easys recall_score:" + str(metrics.recall_score(easys_true_label, easys_pred_label)))
        print("easys f1_score: " + str(metrics.f1_score(easys_true_label, easys_pred_label)))
        print("--------------------------------------------")
        print("hards:")
        print("--------------------------------------------")
        print("hards accuracy_score:" + str(metrics.accuracy_score(hards_true_label, hards_true_label)))
        print("hards precision_score: " + str(metrics.precision_score(hards_true_label, hards_pred_label)))
        print("hards recall_score: " + str(metrics.recall_score(hards_true_label, hards_pred_label)))
        print("hards f1_score: " + str(metrics.f1_score(hards_true_label, hards_pred_label)))



