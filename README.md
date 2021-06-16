<p><img src="./docs/gml_logo.jpg" alt="gml logo" title="GML &amp; tracks" /></p>     
--------------------------------------------------------------------------------

# Gradual Machine Learning(GML) framework
English | [简体中文](./README-zh_CN.md)                

gml is a Python package that provides for Gradual Machine Learning


  - [Introuduction](#introuduction)
  - [Installation](#installation)
  - [Usage](#usage)
  - [API](#api)
  - [FAQ](#faq)
  - [Contributing](#contributing)
  - [Related Efforts](#related-efforts)
  - [The Team](#the-team)
  - [License](#license)
 

## Introuduction
   ### Goal
   In order to support more and more applications of gradual machine learning and help researchers quickly complete model deployment and testing, this project aims to develop a general gradual machine learning platform. Gradual machine learning mainly includes three major steps: easy instance labeling, feature extraction and influence modeling, and gradual inference. According to the different steps of gradual machine learning, multiple unified algorithms are designed and implemented. Currently, several types of factor graphs such as single-factor non-parameterize, binary-factor non-parameterize, and single-factor parameterize are supported, and support factor graph parameter learning algorithm based on stochastic gradient descent and batch gradient descent
   ### Flowchat
   <p><img src="./docs/flowchat.jpg" alt="gml flowchat" title="flowchat &amp; tracks" /></p>    


## Usage
 Before using this framework, you need to prepare your data according to the following [Data structure description](./docs/data_structures.md) .

After preparing the data, you can use this framework as follows.
First you need to prepare a configuration file [example](./examples/example.config),and set some [parameters](./docs/parameter_description.md)
``` python 
[para]
top_m = 2000  
top_k = 10
top_n = 1
update_proportion = -1   
optimization_threshold = -1
balance = False
learning_epoches = 500
inference_epoches = 500
learning_method = sgd
n_process = 1
out = False
```    
Then, call GML as follows:
  ```python            
  with open('variables.pkl', 'rb') as v:
      variables = pickle.load(v)
  with open('features.pkl', 'rb') as f:
      features = pickle.load(f)
  graph = GML.initial("alsa.config", variables, features)
  graph.inference()
```               
Here is an [example](examples/example.py) you can refer.
## API
<details>
    <summary>Easy Instance Labeling</summary>
  
* [EasyInstanceLabeling](./docs/easy_instance_labeling.md "Perform simple instance labeling based on simple user-specified rules or existing unsupervised learning techniques")
    * [label_easy_by_file](./docs/easy_instance_labeling.md "Mark Easy in variables according to the provided easy list")
</details>

<details>
  <summary>Influence Modeling</summary>

* [class EvidentialSupport](./docs/evidential_support.md "Evidence-supported method set")
    * [get_unlabeled_var_feature_evi](./docs/evidential_support.md "Calculate the ratio of 0 to 1 in the evidence variable associated with each unary feature of each hidden variable, and the variable id at the other end of the binary feature")
    * [separate_feature_value](./docs/evidential_support.md "Select the easy feature value of each feature for linear regression")
    * [influence_modeling](./docs/evidential_support.md "Perform linear regression on the updated feature, save all the results of the regression back to the feature, the key is 'regression' ")
    * [init_tau_and_alpha](./docs/evidential_support.md "Calculate tau and alpha parameters for a given feature")
    * [get_dict_rel_acc](./docs/evidential_support.md "Calculate the accuracy of different types of relationships")
    * [construct_mass_function_for_propensity](./docs/evidential_support.md "Build mass function for Evidential Support calculation in no para factor")
    * [labeling_propensity_with_ds](./docs/evidential_support.md "Different types of evidences are combined in different ways for aspect-level sentiment analysis")
* [class Regression](./docs/evidential_support.md "Linear regression related class, perform linear regression on all features, used for the essential support calculation of Entity Resolution")
    * [perform](./docs/evidential_support.md "Perform linear regression method ")
</details>

<details>
  <summary>Gradual Inference</summary>

* [class GML](./docs/gml.md "Progressive machine learning main process")
    * [evidential_support](./docs/gml.md "Calculation essential support")
    * [approximate_probability_estimation](./docs/gml.md "Calculate approximate probability")
    * [select_top_m_by_es](./docs/gml.md "Select the top m hidden variables according to the calculated Evidential Support (large to small)")
    * [select_top_k_by_entropy](./docs/gml.md "Calculate entropy, select top_k hidden variables with small entropy")
    * [select_evidence](./docs/gml.md "Select the edges, variables and features needed for subsequent subgraph construction")
    * [construct_subgraph](./docs/gml.md "After selecting topk hidden variables, create a subgraph")
    * [inference_subgraph](./docs/gml.md "Inference subgraph")
    * [label](./docs/gml.md "Compare the entropy of k hidden variables, select the one with the smallest entropy and label it, and write the parameters learned from this graph back to self.features")
    * [inference](./docs/gml.md "Main process")
    * [score](./docs/gml.md "Calculate the accuracy rate, precision rate, recall rate, f1 value of inference results, etc.")
* [gml_utils](./docs/gml_utils.md "全局函数集合")
    * [load_easy_instance_from_file](./docs/gml_utils.md "Load easy from csv file")
    * [separate_variables](./docs/gml_utils.md "Divide variables into evidence variables and hidden variables")
    * [init_evidence_interval](./docs/gml_utils.md "Initial evidence interval")
    * [init_evidence](./docs/gml_utils.md "Initialize the witness_interval and witness_count attributes of all features")
    * [update_evidence](./docs/gml_utils.md "Because each featurew maintains the evidence_interval attribute, this attribute needs to be updated after each variable is marked")
    * [entropy](./docs/gml_utils.md "Calculate entropy after given probability")
    * [open_p](./docs/gml_utils.md "Weight calculation formula")
    * [combine_evidences_with_ds](./docs/gml_utils.md "Aggregate evidence from different sources")
* [class ApproximateProbabilityEstimation](./docs/approximate_probability_estimation.md "Set of methods for approximate probability calculation")
    * [init_binary_feature_weight](./docs/approximate_probability_estimation.md "Set initial value of binary feature weight")
    * [labeling_conflict_with_ds](./docs/approximate_probability_estimation.md "Evidence support based on (D-S) theory measurement")
    * [get_pos_prob_based_relation](./docs/approximate_probability_estimation.md "Calculate the proportion of positive instances in marked instances with a feature")
    * [construct_mass_function_for_confict](./docs/approximate_probability_estimation.md "Evidence support for calculating each feature connected to an unlabeled variable")
    * [approximate_probability_estimation](./docs/approximate_probability_estimation.md "Calculate the approximate probability of the selected topm hidden variables, used to select topk")
* [class EvidenceSelect](./docs/evidence_select.md "Select evidence variables for latent variable reasoning")
    * [select_evidence](./docs/evidence_select.md "Select evidence variables for latent variable reasoning")
    * [select_evidence_by_custom](./docs/evidence_select.md "Select evidence variables for hidden variable reasoning, user-defined selection method")
* [construct_subgraph](./docs/construct_subgraph.md "Construct factor graph")
    * [construct_subgraph](./docs/construct_subgraph.md "A unified method for constructing factor graphs")
    * [construct_subgraph_for_custom](./docs/construct_subgraph.md "Construction factor graph, user-defined construction method")
</details>
 
## FAQ
  FAQ
## Contributing
  We appreciate all contributions. If you are planning to contribute back bug-fixes, please do so without any further discussion.
  If you plan to contribute new features, utility functions or extensions to the core, please first open an issue and discuss the feature with us
## Related Efforts
  ### [Gradual Machine Learning for Entity Resolution](https://github.com/gml-explore/GML_for_ER)   
  ### [Gradual Machine Learning for Aspect-level Sentiment Analysis](https://github.com/gml-explore/GML_for_ALSA) 


## The Team
> [@Anqi4869](https://github.com/Anqi4869)                        
> [@buglesxu](https://github.com/buglesxu)                      
> [@chenyuWang](https://github.com/DevelopingWang)                 
> [@hxlnwpu](https://github.com/hxlnwpu)                  
> [@zhanghan97](https://github.com/zhanghan97)  
## License
  [Apache License 2.0](LICENSE)

