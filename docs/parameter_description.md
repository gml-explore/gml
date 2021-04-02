# Parameter Description

The following parameters are all set to default values and can be modified selectively.

**top_m**  
```                   
  After the calculation of evidence support is completed, the latent variables with the largest support of the first m evidence are selected as candidates, and their approximate probability and approximate entropy are calculated. The default setting is 2000.
```              
**top_k** 
```                   
  After the approximate entropy of the m hidden variables is calculated, the first k hidden variables with the smallest approximate entropy are selected to construct k subgraphs. The default setting is 10.
```                      
**top_n** 
```                   
 After the k subgraphs are inferred, the entropy is calculated according to the actual inference probability, and the first n variables with the smallest entropy are selected for labeling. The default setting is 1.
```                       
**update_proportion**   
```                   
  In order to speed up the reasoning, the evidence support is recalculated only after the hidden variables of the update_proportion ratio are marked. The default setting is 0.01.
```                
**optimization_threshold**  
```                   
  In order to speed up the reasoning, latent variables whose approximate entropy is less than or equal to optimization_threshold are directly marked and no longer reasoned. The default setting is 0, and a negative value means that this optimization is not required.
```                         
**balance**    
```                   
  When creating a subgraph, the 0-1 variables are balanced. The default setting is False.
```                     
**learning_epoches**  
```                   
  The number of parameter learning rounds, the default setting is 500.
```                    
**inference_epoches**   
```                   
  The number of inference rounds for factor graphs, the default setting is 500.
```                   
**learning_method**    
```                   
  The parameter learning method currently supports both stochastic gradient descent (sgd) and batch gradient descent (bgd). The default setting is sgd.
```                  
**n_process**   
```                   
  The number of multi-process acceleration processes, the default setting is 1.
```                    
**out**     
```                   
  Whether it is necessary to output the probability and label of hidden variable inference to the file in real time, the default is False.
```               