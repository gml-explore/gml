# the class for selecting evidence
> class **EvidenceSelect**(variables, features, balance, interval_evidence_count = 10,subgraph_limit_num,k_hop)[[source]](../evidence_select.py)  
the class for selecting evidence, select evidential variables for latent variables, including two methods for selecting evidential variables, and allow users to define their own methods
>Parameters: 
> - **variables** – One of the input data of gml, the variable of the factor graph 
> - **features** –  One of the input data of gml, the feature of factor graph 
> - **interval_evidence_limit** – When dividing interval sampling evidence, the number of evidences sampled in each interval 
> - **subgraph_limit_num** –  Maximum number of variables allowed in the subgraph 
> - **each_feature_evidence_limit** –  When sampling randomly, the number of evidence for each single factor sampling 

This class currently provides the following methods：
1. evidence_select(var_id)[[source]](../evidence_select.py)
   >Function: Provide a unified evidence selection method, which can be used to construct factor graphs containing parameterized single factor, non-parameterized single factor, and double factor. In this case, call this function.
   >Parameters：  
   > · var_id - the id of latent variable  
   >Return：connected_var_set, connected_edge_set, connected_feature_set  
   >Return type：set