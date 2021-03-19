# the class of gml tools

class **gml_utils** [[source]](../gml_utils.py)         
The set of methods for approximate probability calculation  


This class currently provides the following methods:  
 
1. load_easy_instance_from_file(filename) [[source]](../gml_utils.py)          

    >Function: load easy from csv file  
    >Parameter:  
    > · filename - the path of csv file    
    >Return: simple instance list  
    >Return type: list

2. separate_variables(variables) [[source]](../gml_utils.py)          

    >Function: Divide variables into evidential variables and latent variables  
    >Parameter:  
    > · variables - variables    
    >Return: evidential variables list, latent variables list  
    >Return type: list

3. init_evidence_interval(evidence_interval_count) [[source]](../gml_utils.py)          

    >Function: Initialize the evidence interval  
    >Parameter:  
    > · evidence_interval_count - the number of intervals    
    >Return: a list containing evidence_interval_count intervals  
    >Return type: list

4. init_evidence(features,evidence_interval,observed_variables_set) [[source]](../gml_utils.py)          

    >Function: Initialize the witness_interval and witness_count attributes of all features  
    >Parameter:  
    > · features - features    
    > · evidence_interval - evidential interval    
    > · observed_variables_set - the set of evidential varibles    

5. write_labeled_var_to_evidence_interval(variables,features,var_id,evidence_interval) [[source]](../gml_utils.py)          

    >Function: Because each featurew maintains the evidence_interval attribute, this attribute needs to be updated after every tag  
    >Parameter:  
    > · variables - variables    
    > · features - features    
    > · var_id - The id of the target variable  
    > · evidence_interval - evidential interval     

6. entropy(probability) [[source]](../gml_utils.py)          

    >Function: calculate entropy after given probability  
    >Parameter:  
    > · probability - One probability or list of probabilities    
    >Return: One entropy or a list of entropies  
    >Return type: floating or list

7. open_p(weight) [[source]](../gml_utils.py)          

    >Function: Calculate approximate probability 
    >Parameter:  
    > · weight - Weight   
    >Return: One entropy or a list of entropies  
    >Return type: floating

8. combine_evidences_with_ds(mass_functions, normalization) [[source]](../gml_utils.py)          

    >Function: Summarize and calculate the evidential value of different sources
    >Parameter:  
    > · mass_functions - ？    
    > · normalization - ？    
    >Return: Summarized the calculated evidential value  
    >Return type: list

