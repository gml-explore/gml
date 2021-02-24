# 数据结构说明     
## 变量

variables=list()   #元素类型为dict:variable                                           
variable                      
{                        
  *由用户提供的属性*:                              
*    *'var_id'*:     变量ID，int类型，从0开始计数                
*    *'is_easy'*:    此变量是否是简单实例，取值为True或者False                  
*    *'is_evidence'*:  此变量是否是证据变量，取值为True或者False  
*    *'label'*:  推理出的标签:0为负，1为正，-1为不知道             
*    *'true_label'*:  此变量的真实标签                                 
*    *'feature_set'*:    此变量拥有的所有feature信息                  
    {      
      feature_id1: [theta1,feature_value1],                  
      feature_id2: [theta2,feature_value2],                     
      ...                      
    }   

  *代码运行期间可能会自动生成的属性*        
*    *'inferenced_probability'*: 推理出的概率
*    *'probability'*:   推理出的概率           
*    *'evidential_support'*: 证据支持
*    *'entropy'*: 熵
*    *'approximate_weight'*:近似权重
*    *'approximate_probability'*: 近似概率     
  ...              
            
}

## 特征    

features  = list()      #元素类型为dict:feature         
feature             
{                     
  *由用户提供的属性*
*    *'feature_id'*: 此特征的id， int类型，从0开始计数       
*    *'feature_type'*: 此特征是单因子特征还是双因子特征，目前支持unary_feature和binary_feature两种            
*    *'feature_name'*: 特征名，可选
*    *'parameterize'* : 特征是否函数化（0或1）
*    *'alpha_bound'*:[bound0,bound1] alpha的上下界  
*    *'tau_bound'*:[bound0,bound1] tau的上下界                        
*    *'weight'*:  此特征的所有相关变量信息                                                                 
    {            
      var_id1:        [weight_value1,feature_value1],                         
     (varid3,varid4): [weight_value2,feature_value2],                               
      ...                
    }    

*代码运行期间可能会自动生成的属性*
*    *'tau'*: tau值
*    *'alpha'*:alpha值
*    *'regerssion'*： 线性回归结果               
            
}

