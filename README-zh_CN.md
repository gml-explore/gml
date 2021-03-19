<p><img src="./docs/gml_logo.jpg" alt="gml logo" title="GML &amp; tracks" /></p>     
--------------------------------------------------------------------------------

# 渐进机器学习(GML)框架
简体中文 | [English](./README.md)                

gml是为渐进机器学习提供的一个python模块
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
   ### 开发目的
   渐进机器学习主要包括三个模块：简单实例识别、特征提取与影响力建模和渐进推理。针对不同的领域任务，简单实例识别和特征提取与影响力建模技术差异较大，针对此我们将为其设置通用的接口供用户使用。而渐进推理是渐进机器学习框架中最为重要的模块，其核心原理是所有任务共享的，因而我们将其封装成可直接调用的接口，并将代码开源在GitHub上供用户参考使用。基于研发的渐进机器学习平台，用户只需将数据按照设定的接口生成，并且提供相应的约束条件，便可利用渐近机器学习框架进行推理。具体地，用户需根据不同的任务，首先提供特定的简单实例标注和特征提取技术；然后调用渐进推理模块直接进行推理。需要说明的是，渐进推理模块包含三个步骤：证据支持度估计、熵近似计算和推理子图构建，它们可能会因任务的不同而有所差异，因而若开源项目提供方法无法满足用户需求时，用户可自定义相关方法。针对渐近机器学习开源平台的研发，我们以斯坦福大学发布的开源因子图推理工具Numbskull为基础，结合实体识别与情感分析的不同需求，对 Numbskull源码进行相关修改。原版Numbskull只支持因子单一参数的推理，修改后的Numbskull可支持因子的参数函数化。研发的渐近机器学习开源平台将可作为渐进推理的核心工具，完成参数的学习。
   ###  gml原理
   渐进机器学习框架，借鉴了人类由易到难的学习模式，先由机器自动完成任务中简单实例的标注，然后基于因子图推理渐进式地完成较难实例的标注。不同于深度学习，渐进机器学习无需独立同分布假设，只需很少甚至无需人工标注数据。渐进机器学习框架包括三个模块：简单实例识别、关键特征提取与影响力建模和渐进推理。
   #### 简单实例识别
   给定一个分类任务，如果没有足够的训练数据，通常很难准确标记任务中的所有实例。但是，如果我们只需要简单标注任务中的简单实例，那么情况会变得容易。在实际场景中，可以简单的根据用户指定规则或现有的无监督学习技术来执行简单的实例标记。例如，无监督聚类。渐进机器学习始于简单实例标签的标注结果。因此，简单实例的高精度标注对渐进机器学习在给定任务中的最终性能至关重要。
   #### 关键特征提取与影响力建模
   特征是简单实例与复杂实例之间传递信息的媒介。在这一步中，我们需要提取已标记的简单实例与未标记的复杂实例所共享的特征。为了促进实例之间有效信息的传递，我们提取不同种类的特征以捕获尽可能多的信息。对于提取的特征，我们需要进一步根据其对实例标注的影响力建模。
   #### 渐进推理
   在这一步，我们会逐步标记复杂实例。通过迭代因子图推断对因子图进行参数学习。在每次迭代中，我们选择具有最高证据确定性程度的未标记实例进行标记。重复进行迭代，直到标记了任务中的所有实例为止。值得注意的是，在渐进推理中，当前迭代中新标记的实例将用作后续迭代中的证据。
   ### 框架流程图
   <p><img src="./docs/flowchat.jpg" alt="gml flowchat" title="flowchat &amp; tracks" /></p>    




## 安装
    pip install gml
## 使用
 在使用此框架之前，您需要按照以下[数据结构](./docs/data_structures-zh_CN.md)要求准备您的数据。    
 
在准备完数据之后，您可以按照以下方式使用此框架。
首先需要准备一个配置文件，对一些参数和超参数进行设置。
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
Here is an [example](examples/alsa_example.py) you can refer.

## 接口

<details>
    <summary>Easy Instance Labeling</summary>
  
* [EasyInstanceLabeling](./docs/easy_instance_labeling-zh_CN.md "根据简单的用户指定规则或现有的无监督学习技术来执行简单的实例标记")
    * [label_easy_by_file](./docs/easy_instance_labeling-zh_CN.md "根据提供的easy列表标出variables中的Easy")
    * [label_easy_by_clustering](./docs/easy_instance_labeling-zh_CN.md "通过聚类进行简单实例标注")
    * [label_easy_by_custom](./docs/easy_instance_labeling-zh_CN.md "通过用户自定义进行简单实例标注")
</details>

<details>
  <summary>Influence Modeling</summary>

* [class EvidentialSupport](./docs/evidential_support-zh_CN.md "证据支持计算的方法集合")
    * [get_unlabeled_var_feature_evi](./docs/evidential_support-zh_CN.md "计算每个隐变量的每个unary feature相关联的证据变量里面0和1的比例，以及binary feature另一端的变量id")
    * [separate_feature_value](./docs/evidential_support-zh_CN.md "选出每个feature的easy feature value用于线性回归")
    * [create_csr_matrix](./docs/evidential_support-zh_CN.md "创建稀疏矩阵存储所有variable的所有featureValue，用于后续计算Evidential Support")
    * [influence_modeling](./docs/evidential_support-zh_CN.md "对已更新feature进行线性回归,把回归得到的所有结果存回feature, 键为'regression' ")
    * [init_tau_and_alpha](./docs/evidential_support-zh_CN.md "对给定的feature计算tau和alpha 参数")
    * [evidential_support_by_regression](./docs/evidential_support-zh_CN.md "计算所有隐变量的Evidential Support 参数")
    * [get_dict_rel_acc](./docs/evidential_support-zh_CN.md "计算不同类型关系的准确性")
    * [construct_mass_function_for_propensity](./docs/evidential_support-zh_CN.md "构建mass function，用于Aspect-level情感分析应用中的Evidential Support计算")
    * [labeling_propensity_with_ds](./docs/evidential_support-zh_CN.md "对于不同类型的evidences用不同的方法进行组合，用于Aspect-level情感分析")
    * [evidential_support_by_relation](./docs/evidential_support-zh_CN.md "计算给定隐变量集合中每个隐变量的evidential support,适用于Aspect-level情感分析")
    * [evidential_support_by_custom](./docs/evidential_support-zh_CN.md "用户自定义用于计算evidential support的方法 ")
* [class Regression](./docs/evidential_support-zh_CN.md "线性回归相关类，对所有feature进行线性回归，用于Entity Resolution部分的evidential support计算")
    * [perform](./docs/evidential_support-zh_CN.md "执行线性回归方法，适用于Entity Resolution")
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
    * [write_labeled_var_to_evidence_interval](./docs/gml_utils-zh_CN.md "因为每个featurew维护了evidence_interval属性，所以每标记一个变量之后，需要更新这个属性")
    * [entropy](./docs/gml_utils-zh_CN.md "给定概率之后计算熵")
    * [open_p](./docs/gml_utils-zh_CN.md "权重计算公式")
    * [combine_evidences_with_ds](./docs/gml_utils-zh_CN.md "汇总不同来源的证据")
* [class ApproximateProbabilityEstimation](./docs/approximate_probability_estimation-zh_CN.md "近似概率计算的方法集合")
    * [init_binary_feature_weight](./docs/approximate_probability_estimation-zh_CN.md "设置binary feature权重的初始值")
    * [labeling_conflict_with_ds](./docs/approximate_probability_estimation-zh_CN.md "基于（D-S）理论衡量证据支持")
    * [get_pos_prob_based_relation](./docs/approximate_probability_estimation-zh_CN.md "计算具有某feature的已标记实例中正实例的比例")
    * [construct_mass_function_for_confict](./docs/approximate_probability_estimation-zh_CN.md "计算与某未标记变量相连的每个特征的证据支持")
    * [approximate_probability_estimation](./docs/approximate_probability_estimation-zh_CN.md "计算选出的topm个隐变量的近似概率，用于选topk")
    * [approximate_probability_estimation_by_custom](./docs/approximate_probability_estimation-zh_CN.md "计算选出的topm个隐变量的近似概率，用于选topk,由用户自定义计算规则")
* [class EvidenceSelect](./docs/evidence_select-zh_CN.md "为隐变量推理挑选证据变量")
    * [select_evidence](./docs/evidence_select-zh_CN.md "为隐变量推理挑选证据变量，适用于ER")
    * [select_evidence_by_custom](./docs/evidence_select-zh_CN.md "为隐变量推理挑选证据变量，由用户自定义挑选方法")
* [construct_subgraph](./docs/construct_subgraph-zh_CN.md "构建因子图")
    * [construct_subgraph](./docs/construct_subgraph-zh_CN.md "一种统一的构建因子图的方法")
    * [construct_subgraph_for_mixture](./docs/construct_subgraph-zh_CN.md "构建因子图，适用于ER")
    * [construct_subgraph_for_unaryPara](./docs/construct_subgraph-zh_CN.md "构建因子图，适用于ALSA")
    * [construct_subgraph_for_custom](./docs/construct_subgraph-zh_CN.md "构建因子图，由用户自定义构建方法")
* [how to use numbskull](./docs/how%20to%20use%20numbskull-zh_CN.md "如何使用numbskull")
</details>

## 常见问题解答
 常见问题解答
## 贡献
   我们非常欢迎所有意见，如果您发现了bug,请立即联系我们。如果您想贡献功能更新，请创建一个issuu,经讨论后再提交pull requests。 
## 相关工作
 ### [Gradual Machine Learning for Entity Resolution](https://github.com/gml-explore/GML_for_ER)  
 GML_for_ER主要研究面向实体识别的渐近机器学习的应用关键技术。该项目从任务中可以被机器自动且精确识别的简单实例开始，然后逐渐地通过因子图推理更具挑战的实例，以实现对ER的有效渐进式机器学习。
 ### [Gradual Machine Learning for Aspect-level Sentiment Analysis](https://github.com/gml-explore/GML_for_ALSA) 
 GML_for_ALSA主要研究面向情感分析的渐近机器学习的应用关键技术。该项目针对方面级情感分析，提出一种联合框架SenHint，它利用马尔可夫逻辑网将神经网络的输出结果和语言知识进行集成，并取得相较DNN模型更好的预测性能；针对方面级情感分析的重要子任务——方面识别，提出了一种基于Hint-embedding的神经网络模型，旨在探索句中方面与语义内容之间的关联，并取得显著效果；针对情感词典构建，提出一种无监督的用于构建领域相关词典的算法，其所构建的领域相关词典在面向方面级情感分析的层次注意力机制模型上可取得显著性能。

## 团队介绍
 ### 团队简介  
   本团队为“渐进机器学习算法应用研发团队”，主要研究方向包括：          
  （1）研发渐进机器学习算法在实体识别方面的应用；                
  （2）研发渐进机器学习算法在情感分析方面的应用；                      
  （3）研发通用的渐进机器学习开源平台，支撑面向一般性分类问题的渐进机器学习算法和系统的实现。                       
  该项目启动以来，本团队成员的研究成果在顶级国际期刊和会议TKDE，WWW等获得发表，目前已发表论文共10篇，获得包括国家自然科学基金在内的各类项目2项。  

 ### 项目成员    
> [@Anqi4869](https://github.com/Anqi4869)                        
> [@buglesxu](https://github.com/buglesxu)                      
> [@chenyuWang](https://github.com/DevelopingWang)                 
> [@hxlnwpu](https://github.com/hxlnwpu)                  
> [@zhanghan97](https://github.com/zhanghan97)                 

## 版权
  [Apache License 2.0](LICENSE)

