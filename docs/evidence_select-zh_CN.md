# 证据选择类
> class **EvidenceSelect**(variables, features, balance, interval_evidence_count = 10,subgraph_limit_num,k_hop)[[source]](../evidence_select.py)  
挑选证据类，为隐变量挑选证据变量，包含两种挑选证据变量的方法，且允许用户自定义方法
>参数: 
> - **variables** – gml的输入数据之一，因子图的变量
> - **features** –  gml的输入数据之一，因子图的特征
> - **interval_evidence_count** – 在实体识别任务中，每个feature划分evidence_interval_count个区间，每个区间挑不超过interval_evidence_count个证据变量
> - **subgraph_limit_num** –  在情感分析任务中，子图允许的最大变量个数
> - **k_hop** –  在情感分析任务中，挑选隐变量相邻的证据变量时，距离为k_hop跳的证据变量是相邻变量

此类目前主要提供以下方法：
1. evidence_select(var_id)[[source]](../evidence_select.py)
   >功能：提供一种统一的证据选择方法，可用于构建包含参数化的单因子、非参数化单因子和双因子的因子图，此时调用此函数。  
   >参数：  
   > · var_id - 隐变量id  
   >返回值：connected_var_set, connected_edge_set, connected_feature_set  
   >返回类型：集合

2. select_evidence_for_unary_feature(var_id)[[source]](../evidence_select.py)
   >功能：当需要构建包含参数化的单因子的因子图，此时调用此函数。  
   >参数：  
   > · var_id - 隐变量id  
   >返回值：connected_var_set, connected_edge_set, connected_feature_set  
   >返回类型：集合

3. select_evidence_for_unaryAndBinary_feature(var_id)[[source]](../evidence_select.py)

   >功能：当需要构建包含单因子和双因子的因子图，此时调用此函数。  
   >参数：  
   > · var_id - 隐变量id  
   >返回值：connected_var_set, connected_edge_set, connected_feature_set  
   >返回类型：集合
4. select_evidence_by_custom(var_id)[[source]](../evidence_select.py)

   >功能：用户自定义构建因子图的函数。  
   >参数：  
   > · var_id - 隐变量id 
   >返回类型：集合

