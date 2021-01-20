# 简单实例标注类：  
class EasyInstanceLabeling(variables,features,easys=None) [[source]](../easy_instance_labeling.py)      
功能：根据简单的用户指定规则或现有的无监督学习技术来执行简单的实例标记  
>参数：  
> - **variables** - 变量  
> - **features** - 特征  
> - **easys** - 简单实例，默认值为None  

此类目前提供以下方法：  
1. label_easy_by_file() [[source]](../easy_instance_labeling.py)               

    >功能：根据提供的easy列表标出variables中的Easy，在全局数据结构variables中为元素添加属性'is_easy'（是否为简单实例），'is_evidence'（是否为证据节点）

2. label_easy_by_clustering(easy_proportion) [[source]](../easy_instance_labeling.py)              

    >功能：通过聚类进行简单实例标注  
    >参数：  
    > · easy_proportion - 简单实例占总实例的比例，默认值为0.3

3. label_easy_by_custom() [[source]](../easy_instance_labeling.py)          

    >功能：通过用户自定义进行简单实例标注  