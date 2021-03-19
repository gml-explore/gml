# the main class of gml

class **GML** [[source]](../gml.py)         
The main process of progressive machine learning, which runs through Essential Support, Approximate Probability Estimation, select topm, select topk, construct subgraph, inference subgraph process   


This class currently provides the following methods:  
 
1. evidential_support(variable_set,update_feature_set) [[source]](../gml.py)            

    >Function: calculate essential support  
    >parameter:  
    > · variable_set - the set of target variables    
    > · update_feature_set - the set of target features    

2. approximate_probability_estimation(variable_set) [[source]](../gml.py)            

    >Function: Calculate approximate probability  
    >parameter:  
    > · variable_set - the set of target variables      

3. select_top_m_by_es(m) [[source]](../gml.py)            

    >Function: Select the first m latent variables according to the calculated Evidential Support (large to small)
    >parameter:  
    > · m - The number of latent variables to be selected      
    >Return: a list containing m variable ids  
    >Return type：list

4. select_top_k_by_entropy(var_id_list, k) [[source]](../gml.py)            

    >Function: calculate entropy, select top_k latent variables with small entropy  
    >parameter:  
    > · mvar_id_list - Choose range      
    > · k - The number of latent variables to be selected      
    >Return: a list containing k ids  
    >Return type：list

5. select_evidence(var_id) [[source]](../gml.py)            

    >Function: Select the edges, variables and features which needed for subsequent subgraph construction  
    >parameter:  
    > · var_id - The id of the target variable    
    >Return: Edges, variables and features needed for subsequent subgraph construction  
    >Return type：set

6. construct_subgraph(var_id) [[source]](../gml.py)            

    >Function: Select topk latnet variables and create subgraph  
    >parameter:  
    > · var_id - The id of the target variable    
    >Return: According to the factor graph requirement of numbskull,return weight, variable, factor, fmap, domain_mask, edges  
    >Return type: multiple types

7. inference_subgraph(var_id) [[source]](../gml.py)            

    >Function: inference subgraph  
    >parameter:  
    > · var_id - For entity recognition, var_id is a variable id, and var_id is a set of k variables for sentiment analysis

8. label(var_id_list) [[source]](../gml.py)            

    >Function: Compare the entropy of k latent variables, select the one with the smallest entropy and label it, and write the parameters learned from this graph back to self.features
    >parameter:  
    > · var_id_list - A list of k ids, the probability corresponding to each variable is taken from variables    
    >Return: No output, directly update the label and entropy in vairables, by the way, you can update observed_variables_id and potential_variables_id  
    >Return type：dict

9. inference() [[source]](../gml.py)            

    >Function: Main flow    

10. score() [[source]](../gml.py)            

    >Function: Calculate the accuracy rate, precision rate, recall rate, f1 value of inference results, etc.  
