# gml工具类

class **gml_utils** [[source]](../gml_utils.py)         
近似概率计算的方法集合  


此类目前主要提供以下方法：  
 
1. load_easy_instance_from_file(filename) [[source]](../gml_utils.py)          

    >功能：从csv文件中加载easy  
    >参数：  
    > · filename - csv文件的路径    
    >返回值：简单实例列表  
    >返回类型：列表

2. separate_variables(variables) [[source]](../gml_utils.py)          

    >功能：将variables分成证据变量和隐变量  
    >参数：  
    > · variables - 变量    
    >返回值：证据变量列表，隐变量列表  
    >返回类型：列表

3. init_evidence_interval(evidence_interval_count) [[source]](../gml_utils.py)          

    >功能：初始化证据区间  
    >参数：  
    > · evidence_interval_count - 区间数    
    >返回值：一个包含evidence_interval_count个区间的list  
    >返回类型：列表

4. init_evidence(features,evidence_interval,observed_variables_set) [[source]](../gml_utils.py)          

    >功能：初始化所有feature的evidence_interval属性和evidence_count属性  
    >参数：  
    > · features - 特征    
    > · evidence_interval - 证据区间    
    > · observed_variables_set - 证据变量集合    

5. write_labeled_var_to_evidence_interval(variables,features,var_id,evidence_interval) [[source]](../gml_utils.py)          

    >功能：因为每个featurew维护了evidence_interval属性，所以每标记一个变量之后，需要更新这个属性  
    >参数：  
    > · variables - 变量    
    > · features - 特征    
    > · var_id - 目标变量的id  
    > · evidence_interval - 证据区间    

6. entropy(probability) [[source]](../gml_utils.py)          

    >功能：给定概率之后计算熵  
    >参数：  
    > · probability - 单个概率或者概率列表    
    >返回值：单个熵或者熵的列表  
    >返回类型：浮点数或列表

7. open_p(weight) [[source]](../gml_utils.py)          

    >功能：计算近似概率 
    >参数：  
    > · weight - 权重   
    >返回值：单个熵或者熵的列表  
    >返回类型：浮点数

8. combine_evidences_with_ds(mass_functions, normalization) [[source]](../gml_utils.py)          

    >功能：  汇总计算不同来源的证据值
    >参数：  
    > · mass_functions - ？    
    > · normalization - ？    
    >返回值：汇总计算后的证据值  
    >返回类型：列表

