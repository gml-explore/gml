# 证据选择类
> class **EvidenceSelect**(variables, features, interval_evidence_limit = 200, subgraph_max_num=3000,each_feature_evidence_limit = 2000)[[source]](../evidence_select.py)  
>参数: 
> - **variables** – gml的输入数据之一，因子图的变量
> - **features** –  gml的输入数据之一，因子图的特征
> - **interval_evidence_limit** – 划分区间采样证据时，每个区间采样的证据数目
> - **subgraph_limit_num** –  子图允许的最大变量个数
> - **each_feature_evidence_limit** –  随机采样时，每个单因子采样的证据数目

此类目前主要提供以下方法：
1. evidence_select(var_id)[[source]](../evidence_select.py)
   >功能：统一的证据选择方法，可用于构建包含参数化的单因子、非参数化单因子和双因子的因子图  
   >参数：  
   > · var_id - 隐变量id  
   >返回值：connected_var_set, connected_edge_set, connected_feature_set  
   >返回类型：集合
