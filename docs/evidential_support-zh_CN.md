# 证据支持计算类
class **EvidentialSupport**(variables,features,method,evidence_interval_count,interval_evidence_count) [[source]](../evidential_support.py)                
证据支持计算的方法集合
>参数：
> - **variables** - 变量
> - **features** - 因子
> - **method** - 计算证据支持使用的方法，默认值为'regression'
> - **evidence_interval_count** - 区间数，默认值为10
> - **interval_evidence_count** - 每个区间的变量数，默认值为200

此类目前主要提供以下方法：             
1. get_unlabeled_var_feature_evi()[[source]](../evidential_support.py)
    >功能： 计算每个隐变量的每个unary feature相关联的证据变量里面0和1的比例，以及binary feature另一端的变量id

2. separate_feature_value()[[source]](../evidential_support.py)
    >功能： 选出每个feature的easy feature value用于线性回归

3. create_csr_matrix()[[source]](../evidential_support.py)
    >功能： 创建稀疏矩阵存储所有variable的所有featureValue，用于后续计算Evidential Support
    >返回值：featureValue
    >返回类型：稀疏矩阵

4. influence_modeling()[[source]](../evidential_support.py)
    >功能：对已更新feature进行线性回归,把回归得到的所有结果存回feature, 键为'regression'
    >参数：
    > · update_feature_set - 已更新的feature集合

5. init_tau_and_alpha()[[source]](../evidential_support.py)
    >功能：对给定的feature计算tau和alpha  
    >参数：
    > · feature_set - feature_id的集合

6. computer_unary_feature_es()[[source]](../evidential_support.py)
    >功能：计算所有隐变量的Evidential Support  
    >参数：  
    > · update_feature_set - 已更新的feature集合
    > · variable_set -待计算证据支持的变量集合

7. evidential_support()[[source]](../evidential_support.py)
    >功能：一种通过D-S理论计算所有隐变量的Evidential Support的方法。
    >参数：  
    > · update_feature_set - 已更新的feature集合
    > · variable_set -待计算证据支持的变量集合
8. get_dict_rel_acc()[[source]](../evidential_support.py)
    >功能：计算不同类型关系的准确性

9. construct_mass_function_for_propensity()[[source]](../evidential_support.py)
    >功能：构建mass function，用于非参数化的单因子和双因子连接变量的Evidential Support计算 （Aspect-level情感分析应用中）
    >参数：  
    > · uncertain_degree - 特征的不确定度  
    > · label_prob - 标签匹配的概率,对于词特征来说表示positive实例的比例, 对于关系特征来说表示关系特征的准确率  
    > · unlabel_prob - 标签不匹配的概率 ,对于词特征来说表示negative实例的比例, 对于关系特征来说表示1减去关系特征的准确率
    >返回值 : MassFunction函数  
    >返回类型 : 函数  

10. construct_mass_function_for_para_feature()[[source]](../evidential_support.py)
    >功能：构建mass function，用于参数化的单因子连接变量的Evidential Support计算 （ER应用中）
    >参数：  
    > · theta - 特征的不确定度  
    >返回值 : MassFunction函数  
    >返回类型 : 函数  

11. labeling_propensity_with_ds()[[source]](../evidential_support.py)
    >功能：对于不同类型的evidences用不同的方法进行组合，用于Aspect-level情感分析

12. evidential_support_by_custom()[[source]](../evidential_support.py)
    >功能：用户自定义用于计算evidential support的方法  
    >参数：  
    > · variable_set - 给定隐变量的集合

# 线性回归相关类
class **Regression**(each_feature_easys, n_job, effective_training_count_threshold)[[source]]([source])  
线性回归相关类，对所有feature进行线性回归，用于Entity Resolution部分的evidential support计算
> 参数：
> - **each_feature_easys** - 每个feature拥有的easy变量的feature_value  
> - **n_job** - 计算线程数  
> - **effective_training_count_threshold** - 有效样本数量最小值，默认值为2  

此类目前主要提供以下方法：
1. perform()[[source]](../evidential_support.py)
    >功能：执行线性回归方法，适用于Entity Resolution
