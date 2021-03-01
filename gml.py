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
     GML main process: evidentail support->select_topm->ApproximateProbabilityEstimation->select->topk->inference->label->score
    '''
    def __init__(self,variables, features, learning_method,top_m=2000, top_k=10, top_n = 1,update_proportion= 0.01, balance=False,optimization_threshold = 1e-6,learning_epoches = 1000,inference_epoches = 1000,nprocess=1):
        #check data
        variables_keys= ['var_id','is_easy','is_evidence','true_label','label','feature_set']
        features_keys = ['feature_id','feature_type','parameterize','feature_name','weight']
        learning_methods = ['sgd', 'bgd'] # now support sgd and bgd
        if learning_method not in learning_methods:
            raise ValueError('learning_methods has no this method: '+learning_method)
        for variable in variables:
            for attribute in variables_keys:
                if attribute not in variable:
                    raise ValueError('variables has no key: '+attribute)
        for feature in features:
            for attribute in features_keys:
                if attribute not in feature:
                    raise ValueError('features has no key: '+attribute)
        self.variables = variables
        self.features = features
        self.learning_method = learning_method
        self.labeled_variables_set = set()
        self.top_m = top_m
        self.top_k = top_k
        self.top_n = top_n
        self.optimization_threshold = optimization_threshold
        self.update_proportion = update_proportion
        self.support = EvidentialSupport(variables, features)
        self.select = EvidenceSelect(variables, features)
        self.approximate = ApproximateProbabilityEstimation(variables,features)
        self.subgraph = ConstructSubgraph(variables, features, balance)
        self.learing_epoches = learning_epoches
        self.inference_epoches = inference_epoches
        self.nprocess = nprocess
        self.observed_variables_set, self.poential_variables_set = gml_utils.separate_variables(variables)
        evidence_interval = gml_utils.init_evidence_interval(10)
        gml_utils.init_bound(variables,features)
        gml_utils.init_evidence(features,evidence_interval,self.observed_variables_set)
        #save results
        self.now = str(time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())))
        with open(self.now+'-result.txt', 'w') as f:
            f.write('var_id'+' '+'inferenced_probability'+' '+'inferenced_label'+' '+'ture_label'+'\n')
        #logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - [%(levelname)s]: %(message)s'
        )

    @staticmethod
    def initial(configFile,variables,features):
        '''
        load config from file
        @param configFile:
        @param variables:
        @param features:
        @return:
        '''
        config = ConfigParser()
        config.read(configFile, encoding='UTF-8')
        learning_method = config['para']['learning_method']
        learning_epoches = int(config['para']['learning_epoches'])
        inference_epoches = int(config['para']['inference_epoches'])
        top_m = int(config['para']['top_m'])
        top_k = int(config['para']['top_k'])
        top_n = int(config['para']['top_n'])
        n_process = int(config['para']['n_process'])
        update_proportion = float(config['para']['update_proportion'])
        balance = config['para'].getboolean('balance')
        optimization_threshold = float(config['para']['optimization_threshold'])
        return GML(variables, features, learning_method, top_m, top_k, top_n,update_proportion,balance,optimization_threshold,learning_epoches,inference_epoches,n_process)

    def evidential_support(self, variable_set,update_feature_set):
        '''
        calculate evidential_support
        @param variable_set:
        @param update_feature_set:
        @return:
        '''
        self.support.evidential_support(variable_set,update_feature_set)

    def approximate_probability_estimation(self, variable_set):
        '''
        estimation approximate_probability
        @param variable_set:
        @return:
        '''
        self.approximate.approximate_probability_estimation(variable_set)

    def select_top_m_by_es(self, m):
        '''
        select tom m largest ES poential variables
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
        select top k  smallest entropy poential variables
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

    def evidence_select(self, var_id):
        '''
        select evidence for construct subgraph
        @param var_id:
        @return:
        '''
        connected_var_set, connected_edge_set, connected_feature_set = self.select.evidence_select(var_id)
        return connected_var_set, connected_edge_set, connected_feature_set

    def construct_subgraph(self, var_id):
        '''
        construct_subgraph
        @param var_id:
        @return:
        '''
        evidences = self.evidence_select(var_id)
        weight, variable, factor, fmap, domain_mask, edges_num, var_map,alpha_bound,tau_bound,weight_map_feature,sample_list,wmap,wfactor = self.subgraph.construct_subgraph(evidences,var_id)
        return weight, variable, factor, fmap, domain_mask, edges_num, var_map,alpha_bound,tau_bound,weight_map_feature,sample_list,wmap,wfactor

    def inference_subgraph(self, var_id):
        '''
        subgraph learning and inference
        @param var_id:
        @return:
        '''
        if not type(var_id) == int:
            raise ValueError('var_id should be int' )
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
        weight, variable, factor, fmap, domain_mask, edges_num, var_map,alpha_bound,tau_bound,weight_map_feature,sample_list,wmap,wfactor = self.construct_subgraph(var_id)
        subgraph = weight, variable, factor, fmap, domain_mask, edges_num,alpha_bound,tau_bound,sample_list,wmap,wfactor
        ns_learing.loadFactorGraph(*subgraph)
        # 因子图参数学习
        ns_learing.learning()
        logging.info("subgraph learning finished")
        # 对学习到的参数进行范围控制,并将weight的isfixed属性置为true
        for index, w in enumerate(weight):
            feature_id = weight_map_feature[index]
            w["isFixed"] = True
            if self.features[feature_id]['parameterize'] == 1:
                theta = self.variables[var_id]['feature_set'][feature_id][0]
                x = self.variables[var_id]['feature_set'][feature_id][1]
                a = ns_learing.factorGraphs[0].weight[index]['a']
                b = ns_learing.factorGraphs[0].weight[index]['b']
                w['initialValue'] = theta * a* (x - b)
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

    def label(self,var_id_list):
        '''
        label top_n poential variables
        @param var_id_list:
        @return:
        '''
        entropy_list = list()
        label_list = list()
        for var_id in var_id_list:
            var_index = var_id
            self.variables[var_index]['entropy'] = gml_utils.entropy(self.variables[var_index]['inferenced_probability'])
            entropy_list.append([var_id, self.variables[var_index]['entropy']])
        #如果labelnum小于传进来的变量的个数，就选top_n个
        if len(var_id_list) > self.top_n:
            var = list()
            min_var_list = heapq.nsmallest(self.top_n, entropy_list, key=lambda x: x[1])  # 选出熵最小的变量
            for mv in min_var_list:
                label_list.append(mv[0])
        #否则标记传进来的全部变量
        else:
            label_list = var_id_list
        for var_index in label_list:
            self.variables[var_index]['probability'] = self.variables[var_index]['inferenced_probability']
            self.variables[var_index]['label'] = 1 if self.variables[var_index]['probability'] >= 0.5 else 0
            self.variables[var_index]['is_evidence'] = True
            logging.info('var-' + str(var_index) + " labeled succeed---------------------------------------------")
            self.poential_variables_set.remove(var_index)
            self.observed_variables_set.add(var_index)
            self.labeled_variables_set.add(var_index)
            probability = self.variables[var_index]['probability']
            label = self.variables[var_index]['label']
            true_label = self.variables[var_index]['true_label']
            with open(self.now + '-result.txt', 'a') as f:
                f.write(f'{var_index:7} {probability:10} {label:4} {true_label:4}')
                f.write('\n')
        return label_list


    def inference(self):
        '''
        Through the main process
        @return:
        '''
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
        # If the entropy is less than a certain threshold, mark it directly without reasoning
        if self.optimization_threshold >=0 and self.optimization_threshold <1:
            with open(self.now+'-result.txt', 'a') as f:
                for vid in self.poential_variables_set:
                    if self.variables[vid]['entropy'] <= self.optimization_threshold:
                        self.variables[vid]['probability'] = self.variables[vid]['approximate_probability']
                        self.variables[vid]['is_evidence'] = True
                        self.variables[vid]['label'] = 1 if  self.variables[vid]['probability'] >= 0.5 else 0
                        gml_utils.update_evidence(self.variables, self.features, var, self.support.evidence_interval)
                        logging.info('var-'+str(vid)+' labeled succeed---------------------------------------------')
                        probability = self.variables[vid]['probability']
                        label = self.variables[vid]['label']
                        true_label = self.variables[vid]['true_label']
                        f.write(f'{vid:7} {probability:10} {label:4} {true_label:4}')
                        f.write('\n')
                        self.labeled_variables_set.add(vid)
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
            add_list = [x for x in k_list if x not in inferenced_variables_id]
            if len(add_list) > 0:
                if (self.nprocess == 1):
                    for var_id in add_list:
                    #if var_id not in inferenced_variables_id:
                        self.inference_subgraph(var_id)
                        # 每轮更新期间推理过的变量，因为参数没有更新，所以无需再进行推理。
                        inferenced_variables_id.add(var_id)
                else:
                    futures = []
                    for var_id in add_list:
                        future = pool.submit(self.inference_subgraph, var_id)
                        futures.append(future)
                        inferenced_variables_id.add(var_id)
                    for ft in futures:
                        self.variables[ft.result()[0]]['inferenced_probability'] = ft.result()[1]
            label_list = self.label(k_list)
            gml_utils.update_evidence(self.variables, self.features, var, self.support.evidence_interval)
            gml_utils.update_bound(self.variables,self.features,var)  # 每标记一个变量之后更新上下界
            labeled_var += len(label_list)
            labeled_count += len(label_list)
            logging.info("label_count=" + str(labeled_count))
        #output results
        self.score()

    def save_model(self):
        '''
        save models
        @return:
        '''
        with open(self.now+"_variables.pkl",'wb') as v:
            pickle.dump(self.variables,v)
        with open(self.now+"_features.pkl",'wb') as v:
            pickle.dump(self.features,v)

    def score(self):
        '''
        calculate metrics
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
        print("hards accuracy_score:" + str(metrics.accuracy_score(hards_true_label, hards_pred_label)))
        print("hards precision_score: " + str(metrics.precision_score(hards_true_label, hards_pred_label)))
        print("hards recall_score: " + str(metrics.recall_score(hards_true_label, hards_pred_label)))
        print("hards f1_score: " + str(metrics.f1_score(hards_true_label, hards_pred_label)))



