<p><img src="./docs/gml_logo.jpg" alt="gml logo" title="GML &amp; tracks" /></p>     
--------------------------------------------------------------------------------

# 渐进机器学习(GML)框架
简体中文 | [English](./README.md)                

gml是为渐进机器学习提供的一个python库。
  - [介绍](#介绍)
  - [安装](#安装)
  - [使用](#使用)
  - [接口](#接口)
  - [常见问题解答](#常见问题解答)
  - [贡献](#贡献)
  - [相关工作](#相关工作)
  - [团队介绍](#团队介绍)
  - [版权](#版权)
 

## 介绍
   ### 开发目标
   为了支持越来越广泛的渐进机器学习应用，帮助研究人员快速完成模型部署和测试，本项目旨在开发通用的渐进机器学习平台。本渐进机器学习主要包括三大步骤：简单实例标注、特征提取与影响力建模和渐进推理。针对渐进机器学习的不同步骤，设计实现了多个统一的算法，目前支持单因子无函数化，双因子无函数化，单因子函数化等几种类型的因子图，支持基于随机梯度下降和基于批量梯度下降的因子图参数学习算法。

   ### 框架流程图
   <p><img src="./docs/flowchat.jpg" alt="gml flowchat" title="flowchat &amp; tracks" /></p>    

## 使用
 在使用此框架之前，您需要按照以下[数据结构](./docs/data_structures-zh_CN.md)要求准备您的数据。    
 
在准备完数据之后，您可以按照以下方式使用此框架。
首先需要准备一个配置文件[示例](./examples/example.config)，对一些[参数](./docs/parameter_description-zh-CN.md)进行设置。
``` python 
[para]
top_m = 2000
top_k = 10
top_n = 1
update_proportion = -1
optimization_threshold = -1
balance = False
learning_epoches = 500
inference_epoches = 500
learning_method = sgd
n_process = 1
out = False
```    
然后，按照如下方式调用GML:
  ```python            
  with open('variables.pkl', 'rb') as v:
      variables = pickle.load(v)
  with open('features.pkl', 'rb') as f:
      features = pickle.load(f)
  graph = GML.initial("alsa.config", variables, features)
  graph.inference()
```               
Here is an [example](examples/example.py) you can refer.

## 接口

<details>
    <summary>Easy Instance Labeling</summary>
  
* [EasyInstanceLabeling](./docs/easy_instance_labeling-zh_CN.md "根据简单的用户指定规则或现有的无监督学习技术来执行简单的实例标记")
    * [label_easy_by_file](./docs/easy_instance_labeling-zh_CN.md "根据提供的easy列表标出variables中的Easy")

</details>

<details>
  <summary>Influence Modeling</summary>

* [class EvidentialSupport](./docs/evidential_support-zh_CN.md "证据支持计算的方法集合")
    * [get_unlabeled_var_feature_evi](./docs/evidential_support-zh_CN.md "计算每个隐变量的每个unary feature相关联的证据变量里面0和1的比例，以及binary feature另一端的变量id")
    * [separate_feature_value](./docs/evidential_support-zh_CN.md "选出每个feature的easy feature value用于线性回归")
    * [influence_modeling](./docs/evidential_support-zh_CN.md "对已更新feature进行线性回归,把回归得到的所有结果存回feature, 键为'regression' ")
    * [init_tau_and_alpha](./docs/evidential_support-zh_CN.md "对给定的feature计算tau和alpha 参数")
    * [get_dict_rel_acc](./docs/evidential_support-zh_CN.md "计算不同类型关系的准确性")
    * [construct_mass_function_for_propensity](./docs/evidential_support-zh_CN.md "构建mass function，用于非参数化因子中的Evidential Support计算")
    * [labeling_propensity_with_ds](./docs/evidential_support-zh_CN.md "对于不同类型的evidences用不同的方法进行组合")
    * [evidential_support_by_custom](./docs/evidential_support-zh_CN.md "用户自定义用于计算evidential support的方法 ")
* [class Regression](./docs/evidential_support-zh_CN.md "线性回归相关类，对所有feature进行线性回归，用于Entity Resolution部分的evidential support计算")
    * [perform](./docs/evidential_support-zh_CN.md "执行线性回归方法")
</details>

<details>
  <summary>Gradual Inference</summary>

* [class GML](./docs/gml-zh_CN.md "渐进机器学习主流程")
    * [evidential_support](./docs/gml-zh_CN.md "计算evidential support")
    * [approximate_probability_estimation](./docs/gml-zh_CN.md "计算近似概率")
    * [select_top_m_by_es](./docs/gml-zh_CN.md "根据计算出的Evidential Support(从大到小)选前m个隐变量")
    * [select_top_k_by_entropy](./docs/gml-zh_CN.md "计算熵，选出top_k个熵小的隐变量")
    * [select_evidence](./docs/gml-zh_CN.md "挑选后续建子图需要的边，变量和特征")
    * [construct_subgraph](./docs/gml-zh_CN.md "在选出topk个隐变量之后建立子图")
    * [inference_subgraph](./docs/gml-zh_CN.md "推理子图")
    * [label](./docs/gml-zh_CN.md "比较k个隐变量的熵，选熵最小的一个打上标签，并把此图学习出的参数写回self.features")
    * [inference](./docs/gml-zh_CN.md "主流程")
    * [score](./docs/gml-zh_CN.md "计算推理结果的准确率，精确率，召回率，f1值等")
* [gml_utils](./docs/gml_utils-zh_CN.md "全局函数集合")
    * [load_easy_instance_from_file](./docs/gml_utils-zh_CN.md "从csv文件中加载easy")
    * [separate_variables](./docs/gml_utils-zh_CN.md "将variables分成证据变量和隐变量")
    * [init_evidence_interval](./docs/gml_utils-zh_CN.md "初始化证据区间")
    * [init_evidence](./docs/gml_utils-zh_CN.md "初始化所有feature的evidence_interval属性和evidence_count属性")
    * [update_evidence](./docs/gml_utils-zh_CN.md "因为每个featurew维护了evidence_interval属性，所以每标记一个变量之后，需要更新这个属性")
    * [entropy](./docs/gml_utils-zh_CN.md "给定概率之后计算熵")
    * [open_p](./docs/gml_utils-zh_CN.md "权重计算公式")
    * [combine_evidences_with_ds](./docs/gml_utils-zh_CN.md "汇总不同来源的证据")
* [class ApproximateProbabilityEstimation](./docs/approximate_probability_estimation-zh_CN.md "近似概率计算的方法集合")
    * [init_binary_feature_weight](./docs/approximate_probability_estimation-zh_CN.md "设置binary feature权重的初始值")
    * [labeling_conflict_with_ds](./docs/approximate_probability_estimation-zh_CN.md "基于（D-S）理论衡量证据支持")
    * [get_pos_prob_based_relation](./docs/approximate_probability_estimation-zh_CN.md "计算具有某feature的已标记实例中正实例的比例")
    * [construct_mass_function_for_confict](./docs/approximate_probability_estimation-zh_CN.md "计算与某未标记变量相连的每个特征的证据支持")
    * [approximate_probability_estimation](./docs/approximate_probability_estimation-zh_CN.md "计算选出的topm个隐变量的近似概率，用于选topk")
* [class EvidenceSelect](./docs/evidence_select-zh_CN.md "为隐变量推理挑选证据变量")
    * [select_evidence](./docs/evidence_select-zh_CN.md "为隐变量推理挑选证据变量")
    * [select_evidence_by_custom](./docs/evidence_select-zh_CN.md "为隐变量推理挑选证据变量，由用户自定义挑选方法")
* [construct_subgraph](./docs/construct_subgraph-zh_CN.md "构建因子图")
    * [construct_subgraph](./docs/construct_subgraph-zh_CN.md "一种统一的构建因子图的方法")
    * [construct_subgraph_for_custom](./docs/construct_subgraph-zh_CN.md "构建因子图，由用户自定义构建方法")
</details>

## 常见问题解答
 常见问题解答
## 贡献
   我们非常欢迎所有意见，如果您发现了bug,请立即联系我们。如果您想贡献功能更新，请创建一个issuu,经讨论后再提交pull requests。 
## 相关工作
 ### [Gradual Machine Learning for Entity Resolution](https://github.com/gml-explore/GML_for_ER)  
 ### [Gradual Machine Learning for Aspect-level Sentiment Analysis](https://github.com/gml-explore/GML_for_ALSA) 

## 团队介绍
 ### 团队简介  
   本团队为“渐进机器学习算法应用研发团队”，主要研究方向包括：          
  （1）研发渐进机器学习算法理论体系；                
  （2）研发渐进机器学习算法具体应用；                      
  （3）研发通用的渐进机器学习开源平台，支撑面向一般性分类问题的渐进机器学习算法和系统的实现。                       

 ### 项目成员    
> [@Anqi4869](https://github.com/Anqi4869)                        
> [@buglesxu](https://github.com/buglesxu)                      
> [@chenyuWang](https://github.com/DevelopingWang)                 
> [@hxlnwpu](https://github.com/hxlnwpu)                  
> [@zhanghan97](https://github.com/zhanghan97)                 

## 版权
  [Apache License 2.0](LICENSE)

