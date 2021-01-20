# gml主类

class **GML** [[source]](../gml.py)         
渐进机器学习主流程，贯穿Evidential Support，Approximate Probability Estimation，select topm,select topk,construct subgraph,inference subgraph流程   


此类目前主要提供以下方法：  
 
1. evidential_support(variable_set,update_feature_set) [[source]](../gml.py)            

    >功能：计算evidential support  
    >参数：  
    > · variable_set - 目标变量的集合    
    > · update_feature_set - 目标特征的集合    

2. approximate_probability_estimation(variable_set) [[source]](../gml.py)            

    >功能：计算近似概率  
    >参数：  
    > · variable_set - 目标变量的集合      

3. select_top_m_by_es(m) [[source]](../gml.py)            

    >功能：根据计算出的Evidential Support(从大到小)选前m个隐变量  
    >参数：  
    > · m - 需要选出的隐变量的个数      
    >返回值：一个包含m个变量id的列表  
    >返回类型：列表

4. select_top_k_by_entropy(var_id_list, k) [[source]](../gml.py)            

    >功能：计算熵，选出top_k个熵小的隐变量  
    >参数：  
    > · mvar_id_list - 选择范围      
    > · k - 需要选出的隐变量的个数      
    >返回值：一个包含k个id的列表  
    >返回类型：列表

5. select_evidence(var_id) [[source]](../gml.py)            

    >功能：挑选后续建子图需要的边，变量和特征  
    >参数：  
    > · var_id - 目标变量的id    
    >返回值：后续建子图需要的边，变量和特征  
    >返回类型：集合

6. construct_subgraph(var_id) [[source]](../gml.py)            

    >功能：选出topk个隐变量之后建立子图  
    >参数：  
    > · var_id - 目标变量的id    
    >返回值：按照numbskull的要求因子图,返回weight, variable, factor, fmap, domain_mask, edges  
    >返回类型：多类型

7. inference_subgraph(var_id) [[source]](../gml.py)            

    >功能：推理子图  
    >参数：  
    > · var_id - 用于实体识别则var_id是一个变量id,用于情感分类，则var_id是k个变量的集合    

8. label(var_id_list) [[source]](../gml.py)            

    >功能：比较k个隐变量的熵，选熵最小的一个打上标签，并把此图学习出的参数写回self.features  
    >参数：  
    > · var_id_list - k个id的列表，每个变量对应的概率从variables中拿    
    >返回值：无输出，直接更新vairables中的label和entropy，顺便可以更新一下observed_variables_id和poential_variables_id  
    >返回类型：字典

9. inference() [[source]](../gml.py)            

    >功能：主流程    

10. score() [[source]](../gml.py)            

    >功能：计算推理结果的准确率，精确率，召回率，f1值等  
