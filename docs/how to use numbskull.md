The factor graph weight learning and inference part of this framework uses [numbskull](https://github.com/HazyResearch/numbskull),In order to support the parameterization of single-factor weights, we have partially modified,To avoid conflicts, name it [numbskull_extend](../numbskull_extend/__main__.py)。The documentation of numbskull itself is not perfect. After extensive testing, we have summarized some of the methods used. The following mainly explains the simple use of numbskull and our modified part.
# numbskull usage
  ## [The main data structure in numbskull](../numbskull_extend/numbskulltypes.py)
**Weight**    -Weight                     
[ isFixed ]：Whether the weight is fixed, that is, whether it is a true value, when it is True, there is no need to learn later                     
[ initialValue ]：The initial value of this weight, which can be initialized randomly if it is not known           

**Variable**   -Variable                    
[ isEvidence ]：Whether this variable is evidence variable。     
[ initialValue ]：The label of this variable can take the value 0 or 1     
[ dataType ]：0 means the variable is bool type, 1 means non-bool type    
[ cardinality ]：Indicates that this variable has several possible values     
[ vtf_offset ]：Offset, no need to assign value when building           

**Factor**    -Factor              
[ factorFunction ]：Factor function type, see the range of values [FACTORS](../numbskull_extend/inference.py)       
[ weightId ]：The id of the weight associated with this factor    
[ featureValue ]：The initial value of this factor can be initialized randomly     
[ arity ]：The degree of this factor, which is the number of connected variables on this factor   
[ ftv_offset ]：Offset, used to find the position of the first variable connected by the nth factor in FactorToVar       

**FactorToVar**  -Factor and variable mapping      
[ vid ]：Variable id    
[ dense_equal_to ] :Don't know yet, you don't need to assign a value when building a graph 
## How to create a factor graph   
When creating a factor graph, you need to generate Weight, Variable, Factor, FactorToVar data structures according to the structure of the graph (the connection relationship between factors and variables) in the actual application. For specific drawing ideas, please refer to [factorgraph_test](../numbskull_extend/create_factorgraph.py).                         
## How to perform weight learning and reasoning on factor graphs      
After creating the factor graph, if you need to learn the weights, use this method [learning()](../numbskull_extend/numbskull.py),If you need to reason about variables, use this method [inference()](../numbskull_extend/numbskull.py).
For details, please refer to [inference_factorgraph](../numbskull_extend/inference_factorgraph.py)


# What have we modified             
To support parameterization of single factors(refer to [GML_for_ER](https://github.com/gml-explore/GML_for_ER)),We added new attributes to weight and FactorToVar.        

 **Weight**  -Weight                       
[ parameterize ]：Whether this weight needs to be parameterized, required for True, not required for False                            
[ a ]：Parameter 1, we used tau in GML for ER                    
[ b ]：Parameter 2, the alpha we used in GML for ER         
                  
 **FactorToVar**  -Factor and variable mapping                      
[ x ]：feature value        
[ theta ]：This factor supports the evidence of this hidden variable. We used theta in GML for ER       

In addition, we modified this method [sample_and_sgd](../numbskull_extend/learning.py),First determine whether a factor needs to be parameterized. If necessary, perform gradient descent on the required parameters. The returned weight value is calculated by this formula: w = theta * tau*(x-alpha)。We also modified this method [potential](../numbskull_extend/inference.py),The weight is also calculated by this formula: w = theta * tau*(x-alpha)。



