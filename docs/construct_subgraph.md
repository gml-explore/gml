# the class for Subgraph construction:
class **ConstructSbugraph**(variables, features, balance) [[source]](../construct_subgraph.py)                 
Factor graph construction class
> Parameters：
> - **variables** – input data of gml, the variable of the factor graph.         
> - **features** –  input data of gml, The characteristics of the factor graph.           
> - **balance** –   Whether balance the labels when constructing the subgraph. If the number of evidence variable labels(the number of 0 and 1)is quite large, it is set to True, otherwise it is set to False.

This class currently provides the following methods:
1. construct_subgraph(evidences)[[source]](../construct_subgraph.py)

    >Function：A unified construct subgraph method,it is necessary to construct a factor graph containing single factors(parameterize or not) and double factors, and this function is called at this time.  
    >Parameters：  
    > · evidences - Variables, edges, and feature sets required when constructing factor graphs  
    >Returns：the data for Numbskull inference（weight, variable, factor, fmap, domain_mask, edges_num, var_map）  
    >Return type：Multiple types
2. construct_subgraph_for_mixture(evidences)[[source]](../construct_subgraph.py)

    >Function：it is necessary to construct a factor graph containing single factors and double factors, and this function is called at this time.  
    >Parameters：  
    > · evidences - Variables, edges, and feature sets required when constructing factor graphs  
    >Returns：the data for Numbskull inference（weight, variable, factor, fmap, domain_mask, edges_num, var_map）  
    >Return type：Multiple types

3. construct_subgraph_for_unaryPara(evidences)[[source]](../construct_subgraph.py)

    >Function：it is necessary to construct a factor graph containing parameterized single factors, and this function is called at this time.  
    >Parameters：  
    > · evidences - Variables, edges, and feature sets required when constructing factor graphs  
    >Returns：the data for Numbskull inference（weight, variable, factor, fmap, domain_mask, edges_num, var_map）  
    >Return type：Multiple types

4. construct_subgraph_for_custom(var_id, evidences)[[source]](../construct_subgraph.py)

    >Function：User-defined functions for constructing factor graphs.  
    >Parameters：  
    > · var_id - variable id  
    > · evidences - Variables, edges, and feature sets required when constructing factor graphs  
