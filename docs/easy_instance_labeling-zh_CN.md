# 简单实例标注类：  
class EasyInstanceLabeling(variables,features) [[source]](../easy_instance_labeling.py)      
功能：根据简单的用户指定规则或现有的无监督学习技术来执行简单的实例标记  
>参数：  
> - **variables** - 变量  
> - **features** - 特征  

此类目前提供以下方法：  
1. label_easy_by_file(easys) [[source]](../easy_instance_labeling.py)               

    >功能：根据提供的easy列表标出variables中的Easy，在全局数据结构variables中为元素添加属性'is_easy'（是否为简单实例），'is_evidence'（是否为证据节点） 
    >参数：  
    >  · easys - easy variable id and label collection           
    >返回值：无            
    >返回类型：无                 