# Data structure description     
## Variables

variables=list()   #The element type is:  dict:variable                                           
variable                      
{                        
  *User-provided attributes*:                              
*    *'var_id'*:     Variable ID, int type, counting from 0                     
*    *'is_easy'*:    Whether this variable is a easy instance, the value is True or False                  
*    *'is_evidence'*:  Whether this variable is an evidence variable, the value is True or False
*    *'label'*:  Inferred labels: 0 is negative, 1 is positive, -1 is unknown               
*    *'true_label'*:  The true label of this variable                                 
*    *'feature_set'*:    All feature information owned by this variable                 
    {      
      feature_id1: [theta1,feature_value1],                  
      feature_id2: [theta2,feature_value2],                     
      ...                      
    }   

  *Attributes that may be automatically generated while the code is running*        
*    *'inferenced_probability'*: Inferred probability
*    *'probability'*:   Inferred probability           
*    *'evidential_support'*: Evidence support
*    *'entropy'*: Entropy
*    *'approximate_weight'*:Approximate weight
*    *'approximate_probability'*: Approximate probability     
  ...              
            
}

## Features    

features  = list()      #The element type is: dict:feature         
feature             
{                     
  *User-provided attributes*
*    *'feature_id'*: The id of this feature, int type, counting from 0      
*    *'feature_type'*: Whether this feature is a single factor feature or a dual factor feature, currently supports both unary_feature and binary_feature            
*    *'feature_name'*: Feature name, optional              
*    *'weight'*:  Information about all relevant variables of this feature                                                               
    {            
      var_id1:        [weight_value1,feature_value1],                         
     (varid3,varid4): [weight_value2,feature_value2],                               
      ...                
    }    

*Attributes that may be automatically generated while the code is running*
*    *'tau'*: tau value
*    *'alpha'*:alpha value
*    *'regerssion'*： Linear regression results         
            
}

