# 子图构建类
class **ConstructSbugraph**(variables, features, balance) [[source]](../construct_subgraph.py)                 
因子图构建类
> 参数：
> - **variables** – gml的输入数据之一，因子图的变量。         
> - **features** –  gml的输入数据之一，因子图的特征。           
> - **balance** –   建子图时是否需要平衡标签，若证据变量标签数目即0和1的数目差距较大，则置为True，否则置为False。

此类目前主要提供以下方法：
1. construct_subgraph(evidences)[[source]](../construct_subgraph.py)

    >功能：一种统一的建图方法，可用于构建包含单因子（参数化或者非参数化）和双因子的因子图。  
    >参数：  
    > · evidences - 构建因子图时所需的变量，边，特征集合  
    >返回值：Numbskull推理所需数据（weight, variable, factor, fmap, domain_mask, edges_num, var_map）  
    >返回类型：多类型
2. construct_subgraph_for_mixture(evidences)[[source]](../construct_subgraph.py)

    >功能：可用于构建包含单因子（非参数化）和双因子的因子图，此时调用此函数。  
    >参数：  
    > · evidences - 构建因子图时所需的变量，边，特征集合  
    >返回值：Numbskull推理所需数据（weight, variable, factor, fmap, domain_mask, edges_num, var_map）  
    >返回类型：多类型

3. construct_subgraph_for_unaryPara(evidences)[[source]](../construct_subgraph.py)

    >功能：可用于构建包含参数化的单因子的因子图，此时调用此函数。  
    >参数：  
    > · evidences - 构建因子图时所需的变量，边，特征集合  
    >返回值：Numbskull推理所需数据（weight, variable, factor, fmap, domain_mask, edges_num, var_map）  
    >返回类型：多类型

4. construct_subgraph_for_custom(var_id, evidences)[[source]](../construct_subgraph.py)

    >功能：用户自定义构建因子图的函数。  
    >参数：  
    > · var_id - 变量id  
    > · evidences - 构建因子图时所需的变量，边，特征集合  
