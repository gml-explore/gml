# the class for computing evidential support
class **EvidentialSupport**(variables,features,method,evidence_interval_count,interval_evidence_count) [[source]](../evidential_support.py)                
the set of computing evidential support 
>Parameters：
> - **variables** - variables
> - **features** - factors
> - **method** - The method of computing evidential support, the default value is 'regression'
> - **evidence_interval_count** - The number of intervals, the default value is 10
> - **interval_evidence_count** - The number of variables in each interval, the default value is 200

This class currently provides the following methods：             
1. get_unlabeled_var_feature_evi()[[source]](../evidential_support.py)
    >Function: Calculate the ratio of 0 to 1 in the evidential variables associated with each unary feature of each latent variable, and the variable id at the other end of the binary feature

2. separate_feature_value()[[source]](../evidential_support.py)
    >Function: Select the easy feature value of each feature for linear regression

3. create_csr_matrix()[[source]](../evidential_support.py)
    >Function: Create a sparse matrix to store featureValue of all variables for subsequent calculation of Evidential Support
    >Return：featureValue
    >Return type：sparse matrix

4. influence_modeling()[[source]](../evidential_support.py)
    >Function: Perform linear regression on the updated feature, save all the results of the regression back to the feature, the key is 'regression'
    >Parameter：
    > · update_feature_set - the set of updated feature

5. init_tau_and_alpha()[[source]](../evidential_support.py)
    >Function: calculate tau and alpha for feature  
    >Parameter：
    > · feature_set - the set of feature_id

6. evidential_support_by_regression()[[source]](../evidential_support.py)
    >Function: Calculate the Evidential Support of all latent variables
    >Parameter：  
    > · update_feature_set - the set of updated feature

7. get_dict_rel_acc()[[source]](../evidential_support.py)
    >Function: Calculate the accuracy of different types of relationships

8. construct_mass_function_for_propensity()[[source]](../evidential_support.py)
    >Function: Build a mass function for Evidential Support calculation in Aspect-level sentiment analysis applications
    >Parameter：  
    > · uncertain_degree - Feature uncertainty  
    > · label_prob - The probability of label matching, which represents the proportion of positive instances for word features and the accuracy of relationship features for relationship features  
    > · unlabel_prob - The probability of label unmatch, which represents the proportion of negative instances for word features and 1 minus the accuracy of relationship features for relationship features
    >Return : MassFunction  
    >Return type : function  

9. labeling_propensity_with_ds()[[source]](../evidential_support.py)
    >Function: Combine different methods for different types of evidences for aspect-level sentiment analysis

10. evidential_support_by_relation()[[source]](../evidential_support.py)
    >Function: calculate the essential support of each latent variable in a given set of latent variables, suitable for Aspect-level sentiment analysis 
    >Parameter：  
    > · variable_set - a set of latent variables

11. evidential_support_by_custom()[[source]](../evidential_support.py)
    >Function: User-defined method for calculating essential support
    >Parameter：  
    > · variable_set - a set of latent variables

# the class of related Linear regression
class **Regression**(each_feature_easys, n_job, effective_training_count_threshold)[[source]]([source])  
The class of related Linear regression, perform linear regression on all features, used for the essential support calculation of Entity Resolution
> Parameter：
> - **each_feature_easys** - The feature_value of the easy variable owned by each feature  
> - **n_job** - calculate the number of threads  
> - **effective_training_count_threshold** - The minimum number of effective samples, the default value is 2  

This class currently provides the following methods:
1. perform()[[source]](../evidential_support.py)
    >Function: Perform linear regression method, suitable for Entity Resolution
