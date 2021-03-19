# the class for Approximate Probability Estimation：  
class **ApproximateProbabilityEstimation**(variables, features, method) [[source]](../approximate_probability_estimation.py)         
Methods for Approximate Probability Estimation class  
>Parameters:
> - **variables** - variables
> - **features** - features
> - **method** - The method used to calculate the approximate probability, the default value is 'relation'

This class currently provides the following methods:  
 
1. init_binary_feature_weight(value1,value2) [[source]](../approximate_probability_estimation.py)          

    >Function：Set initial value of binary feature weight  
    >Parameters:  
    > · value1 - binary feature initial weight value 1  
    > · value2 - binary feature initial weight value 2  
    >Returns：binary feature weight initial value  
    >Return type：dictionary

2. labeling_conflict_with_ds(mass_functions) [[source]](../approximate_probability_estimation.py)               

    >Function：based on (D-S) theory Evidence support measurement  
    >Parameters：  
    > · mass_functions - 
    >Returns：Evidence support value  
    >Return type：float

3. get_pos_prob_based_relation(var_id, weight) [[source]](../approximate_probability_estimation.py)                    

    >Function：Calculate the proportion of positive instances in marked instances with a feature  
    >Parameters：  
    > · var_id - Target variable id  
    > · weight - feature weight  
    >Returns：Proportion of positive instances in marked instances with a feature  
    >Return type：float  

4. construct_mass_function_for_mixture(uncertain_degree, pos_prob, neg_prob) [[source]](../approximate_probability_estimation.py) 

    >Function：Evidence support for calculating each feature connected to an unlabeled variable  
    >Parameters：  
    > · theta - Uncertainty of a feature  
    >Returns：MassFunction function  
    >Return type：function

5. construct_mass_function_for_ER(alpha,tau,featureValue) [[source]](../approximate_probability_estimation.py) 

    >Function：Evidence support for calculating each feature connected to an unlabeled variable  
    >Parameters：  
    > · theta - Uncertainty of a feature  
    > · alpha - Parameters after unaryfactor linear regression
    > · tau - Parameters after unaryfactor linear regression
    > · featureValue - featurevalue
    >Returns：MassFunction function  
    >Return type：function

6. approximate_probability_estimation(variable_set) [[source]](../approximate_probability_estimation.py)          

    >Function：Calculate the approximate probability of the selected topm hidden variables, used to select topk, suitable for ER  
    >Parameters：  
    > · variable_set - Latent variable data set
    
7. approximate_probability_estimation_for_unary_feature(variable_set) [[source]](../approximate_probability_estimation.py)          

    >Function：Calculate the approximate probability of the selected topm hidden variables, used to select topk, suitable for ER  
    >Parameters：  
    > · variable_set - Latent variable data set

8. approximate_probability_estimation_for_unaryAndBinary_feature(variable_set) [[source]](../approximate_probability_estimation.py)       

    >Function：Calculate the approximate probability of the selected topm hidden variables, used to select topk, suitable for ALSA 
    >Parameters：  
    > · variable_set - Latent variable data set

9. approximate_probability_estimation_by_custom(variable_set) [[source]](../approximate_probability_estimation.py)           

    >Function：Calculate the approximate probability of the selected topm hidden variables, used to select topk, user-defined calculation rules  
    >Parameters：  
    > · variable_set - Latent variable data set
