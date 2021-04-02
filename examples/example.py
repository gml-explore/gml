import pickle
import time
import warnings
from easy_instance_labeling import EasyInstanceLabeling
from gml import  GML
from gml_utils import *
from sklearn import metrics


if __name__ == '__main__':
    warnings.filterwarnings('ignore')  
    begin_time = time.time()
    # data preparation 
    dataname = "restaurant15_acsa"
    with open(dataname + '_variables.pkl', 'rb') as v:
        variables = pickle.load(v)
    with open(dataname + '_features.pkl', 'rb') as f:
        features = pickle.load(f)
    #initialize the GML object 
    graph = GML.initial("alsa.config", variables, features)
    #inference
    graph.inference()
    #Output reasoning time 
    end_time = time.time()
    print('Running time: %s Seconds' % (end_time - begin_time))
