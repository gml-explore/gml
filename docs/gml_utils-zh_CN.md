# gml工具包

**gml_utils** [[source]](../gml_utils.py)         

gml常用工具类，目前主要提供以下方法：  
 
1. load_easy_instance_from_file(filename) [[source]](../gml_utils.py)          

    >功能: 从文件中加载easy信息               
    >参数:  
    > · filename - csv文件路径   
    >返回值: 简单实例列表
    >返回类型: 列表       

2. separate_variables(variables) [[source]](../gml_utils.py)          

    >功能: 划分证据变量和隐变量                     
    >参数:  
    > · variables - 变量集合              
    >返回值: 证据变量集合，隐变量集合               
    >返回类型: 集合               

3. init_evidence_interval(evidence_interval_count) [[source]](../gml_utils.py)          

    >功能: 初始化证据区间                   
    >参数:  
    > · evidence_interval_count - 证据区间数目      
    >返回值: 证据区间 evidence_interval            
    >返回类型: 列表             

4. init_evidence(features,evidence_interval,observed_variables_set) [[source]](../gml_utils.py)          

    >功能: 为每个特征添加证据区间属性和证据数目属性                    
    >参数:  
    > · features - 特征集合    
    > · evidence_interval - 证据区间  
    > · observed_variables_set - 证据变量集合    
    >返回值: 无            
    >返回类型: 无             
5. update_evidence(variables,features,var_id_list,evidence_interval)[[source]](../gml_utils.py) 
    
    >功能: 更新每个特征添的证据区间属性和证据数目属性               
    >参数:           
    > · variables - 变量集合 
    > · features - 特征集合    
    > · var_id_list - 需要更新的变量列表  
    > · evidence_interval - 证据区间  
    >返回值: 无              
    >返回类型: 无           
6. init_bound(variables,features)[[source]](../gml_utils.py)
    
    >功能: 初始化参数边界              
    >参数:           
    > · variables - 变量集合 
    > · features - 特征集合            
    >返回值: 无           
    >返回类型: 无            
7. update_bound(variables,features,var_id_list))[[source]](../gml_utils.py)   
    
    >功能: 变量推理完成后，更新参数边界                                   
    >参数:           
    > · variables - 变量集合 
    > · features - 特征集合          
    > · var_id_list - 需要更新的变量列表              
    >返回值: 无             
    >返回类型: 无            

8. entropy(probability) [[source]](../gml_utils.py) 
    
    >功能: 计算给定概率的熵                    
    >参数:  
    > · probability - 单个概率或者概率列表    
    >返回值: 单个熵或者熵列表            
    >返回类型: 单精度浮点数或列表              

9.  open_p(weight) [[source]](../gml_utils.py)          

    >功能: 计算近似概率                      
    >参数:  
    > · weight - 权重        
    >返回值: 近似概率值                
    >返回类型: 单精度浮点数              

10. combine_evidences_with_ds(mass_functions, normalization) [[source]](../gml_utils.py)          

    >功能: 综合一个变量的所有证据支持              
    >参数:  
    > · mass_functions - 质量函数    
    > · normalization - 是否正则化    
    >返回值: 综合后的证据支持                
    >返回类型: 列表                


class **Logger**(object) [[source]](../gml_utils.py)   
 日志类，用于将结果同时输出到文件和控制台，目前主要提供以下方法
>参数: 
> - **object** – 写入的文件对象

1. write(message) [[source]](../gml_utils.py) 
    >功能:  同时写入到文件和控制台
    >参数:  
    > · message - 写入的内容   
    >返回值: 无              
    >返回类型: 无              
