# the class for selecting evidence
> class **EvidenceSelect**(variables, features, balance, interval_evidence_count = 10,subgraph_limit_num,k_hop)[[source]](../evidence_select.py)  
the class for selecting evidence, select evidential variables for latent variables, including two methods for selecting evidential variables, and allow users to define their own methods
>Parameters: 
> - **variables** – One of the input data of gml, the variables of the factor graph
> - **features** –  One of the input data of gml, the features of the factor graph
> - **interval_evidence_count** – In the entity recognition task, for each feature,to divide evidence_interval_count intervals, and select no more than interval_evidence_count evidential variables in each interval 
> - **subgraph_limit_num** –  In the sentiment analysis task, the maximum number of variables allowed in the subgraph
> - **k_hop** –  In the sentiment analysis task, when selecting evidential variables that are adjacent to the latent   variable, the evidential variable whose distance does not exceed k_hop is the adjacent variable

This class currently provides the following methods：
1. evidence_select(var_id)[[source]](../evidence_select.py)
   >Function: Provide a unified evidence selection method, which can be used to construct factor graphs containing parameterized single factor, non-parameterized single factor, and double factor. In this case, call this function.
   >Parameters：  
   > · var_id - the id of latent variable  
   >Return：connected_var_set, connected_edge_set, connected_feature_set  
   >Return type：set

2. select_evidence_for_unary_feature(var_id)[[source]](../evidence_select.py)

   >Function: It is necessary to construct a factor graph containing parameterized unary factors, and this function is called at this time.  
   >Parameters：  
   > · var_id - the id of latent variable  
   >Return：connected_var_set, connected_edge_set, connected_feature_set  
   >Return type：set

3. select_evidence_for_unaryAndBinary_feature(var_id)[[source]](../evidence_select.py)

   >Function：It is necessary to construct a factor graph containing unary factors and binary factors, and this function is called at this time.  
   >Parameters：  
   > · var_id - the id of latent variable
   >Return：connected_var_set, connected_edge_set, connected_feature_set  
   >Return type：set
4. select_evidence_by_custom(var_id)[[source]](../evidence_select.py)

   >Function：User-defined functions for constructing factor graphs.
   >Parameters：  
   > · var_id - the id of latent variable 
   >Return type：set


