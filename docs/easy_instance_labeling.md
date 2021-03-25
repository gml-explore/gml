# the class for Easy Instance Labeling：  
class EasyInstanceLabeling(variables,features,easys=None) [[source]](../easy_instance_labeling.py)      
function：Perform easy instance labeling based on simple user-specified rules or existing unsupervised learning techniques  
>Parameters：  
> - **variables** - variables  
> - **features** - features  
> - **easys** - easy instances, The default value is None  

This class currently provides the following methods:   
1. label_easy_by_file() [[source]](../easy_instance_labeling.py)               

    >Function：label Easy in the variable according to the provided easy list, and add the attribute 'is_easy' （is it a easy instance）to the element in the global data structure variable，'is_evidence'（whether it is an evidence variable）               
    >Parameters：                
    >  · easys -      
    >Return：none             
    >Return type：none      
