本框架的因子图权重学习和推理部分使用了[numbskull](https://github.com/HazyResearch/numbskull),为了支持单因子权重的参数化，我们对其进行了部分修改，为了避免冲突，将其命名为[numbskull_extend](../numbskull_extend/__main__.py)。numbskull本身的文档并不完善，经过大量测试，我们总结了一些使用的方法。以下主要说明numbskull的简单使用和我们修改的部分。
# numbskull使用
  ## [numbskull中主要的数据结构](../numbskull_extend/numbskulltypes.py)
**Weight**  -权重            
[ isFixed ]：此权重是否固定，即是否是真实值，为True时后期不需要再学习                
[ initialValue ]：此权重的初始值，不知道时可随机初始化           

**Variable** -变量     
[ isEvidence ]：此变量是否是证据变量，为1时是，为0时否。     
[ initialValue ]：此变量的标签，可取值0或1     
[ dataType ]：为0时表示此变量是bool型，为1表示非bool型    
[ cardinality ]：表示此变量有几个可能的取值      
[ vtf_offset ]：偏移量，建图时可不赋值           

**Factor** -因子    
[ factorFunction ]：因子函数类型，取值范围见[FACTORS](../numbskull_extend/inference.py)       
[ weightId ]：此因子关联的权重的id     
[ featureValue ]：此因子的初始值，可随机初始化     
[ arity ]：此因子的度，即此因子上连接变量的个数    
[ ftv_offset ]：偏移量，用来在FactorToVar找第n个因子连接的第一个变量的位置          

**FactorToVar**  -因子和变量的映射      
[ vid ]：变量的id    
[ dense_equal_to ] :暂不知，建图时可不赋值 
## 如何创建因子图   
创建因子图时，需要根据实际应用中图的结构（因子和变量的连接关系）来生成Weight，Variable，Factor， FactorToVar这几个数据结构。具体的建图思路可参考[factorgraph_test](../numbskull_extend/create_factorgraph.py)               
## 如何对因子图进行权重学习和推理      
创建完因子图之后，如果需要对权重进行学习，则使用[learning()](../numbskull_extend/numbskull.py)方法,如果需要对变量进行推理，则使用[inference()](../numbskull_extend/numbskull.py)方法。
具体细节可参考[inference_factorgraph](../numbskull_extend/inference_factorgraph.py)


# 我们修改了什么       
为了支持单因子的参数化(参见[GML_for_ER](https://github.com/gml-explore/GML_for_ER))，我们在weight和FactorToVar中添加了新的属性。           

 **Weight**  -权重                
[ parameterize ]：此权重是否需要参数化,为True时需要，为False时不需要                    
[ a ]：参数1，我们用在了GML for ER中的tau                 
[ b ]：参数2，我们用在了GML for ER中的alpha     
                  
 **FactorToVar**  -因子和变量的映射                
[ x ]：特征值        
[ theta ]：此因子对此隐变量的证据支持。我们用在了GML for ER中的theta  

此外，我们修改了[sample_and_sgd](../numbskull_extend/learning.py)函数,先判断一个因子是否需要参数化，如果需要，就对所需参数进行梯度下降,返回的权重值由此公式计算得到：w = theta * tau*(x-alpha)。我们还修改了[potential](../numbskull_extend/inference.py)函数，权重同样由此公式计算得来：w = theta * tau*(x-alpha)。



