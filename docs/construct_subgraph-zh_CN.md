# 子图构建类
class **ConstructSbugraph**(variables, features, balance) [[source]](../construct_subgraph.py)                 
因子图构建类
> 参数：
> - **variables** – gml的输入数据之一，因子图的变量。         
> - **features** –  gml的输入数据之一，因子图的特征。           
> - **balance** –   建子图时是否需要平衡标签，若证据变量标签数目即0和1的数目差距较大，则置为True，否则置为False。

此类目前主要提供以下方法：
1. construct_subgraph(evidences)[[source]](../construct_subgraph.py)

    >功能：统一的建图方法，可用于构建包含单因子（参数化或者非参数化）和双因子的因子图。  
    >参数：  
    >  · evidences - 构建因子图时所需的变量，边，特征集合  
    >返回值：Numbskull推理所需数据（weight, variable, factor, fmap, domain_mask, edges_num, var_map, alpha_bound, tau_bound, weight_map_feature, sample_list, wmap, wfactor）  
    >返回类型：多类型