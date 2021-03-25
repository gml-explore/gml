# 近似概率计算类：  
class **ApproximateProbabilityEstimation**(variables, features, method) [[source]](../approximate_probability_estimation.py)         
近似概率计算的方法集合  
>参数：
> - **variables** - 变量
> - **features** - 特征
> - **method** - 计算近似概率使用的方法，默认值为'relation'

此类目前主要提供以下方法：  
 
1. init_binary_feature_weight(value1,value2) [[source]](../approximate_probability_estimation.py)          

    >功能：设置binary feature权重的初始值  
    >参数：  
    > · value1 - binary feature初始权重值1  
    > · value2 - binary feature初始权重值2  
    >返回值：binary feature权重初始值  
    >返回类型：字典

2. labeling_conflict_with_ds(mass_functions) [[source]](../approximate_probability_estimation.py)               

    >功能：基于（D-S）理论衡量证据支持  
    >参数：  
    > · mass_functions -   
    >返回值：证据支持量  
    >返回类型：浮点数

3. get_pos_prob_based_relation(var_id, weight) [[source]](../approximate_probability_estimation.py)                    

    >功能：计算具有某feature的已标记实例中正实例的比例  
    >参数：  
    > · var_id - 目标变量id  
    > · weight - feature的权重  
    >返回值：具有某feature的已标记实例中正实例的比例  
    >返回类型：浮点数  

4. construct_mass_function_for_confict(uncertain_degree, pos_prob, neg_prob) [[source]](../approximate_probability_estimation.py) 

    >功能：构造massfunction,计算与某未标记变量相连的每个特征的证据支持  
    >参数：  
    > · uncertain_degree - 某特征的不确定性  
    > · pos_prob - 对已标记实例的证据支持  
    > · neg_prob - 对未标记实例的证据支持  
    >返回值：MassFunction函数  
    >返回类型：函数

5. construct_mass_function_for_ER(alpha,tau,featureValue) [[source]](../approximate_probability_estimation.py) 

    >功能：构造massfunction,计算与某未标记变量相连的每个特征的证据支持  
    >参数：  
    > · alpha - 针对单因子线性回归后的参数  
    > · tau - 针对单因子线性回归后的参数
    > · featureValue - 特征值
    >返回值：MassFunction函数  
    >返回类型：函数

6. approximate_probability_estimation(variable_set) [[source]](../approximate_probability_estimation.py)       

    >功能：计算选出的topm个隐变量的近似概率，用于选topk,适用于含有非参数化单因子、参数化单因子和双因子的计算  
    >参数：  
    > · variable_set - 隐变量数据集

7. approximate_probability_estimation_by_custom(variable_set) [[source]](../approximate_probability_estimation.py)           

    >功能：计算选出的topm个隐变量的近似概率，用于选topk,由用户自定义计算规则  
    >参数：  
    > · variable_set - 隐变量数据集
