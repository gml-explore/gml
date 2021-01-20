"""TODO."""

from __future__ import print_function, absolute_import
import numpy as np

Meta = np.dtype([('weights', np.int64),
                 ('variables', np.int64),
                 ('factors', np.int64),
                 ('edges', np.int64)])


Weight = np.dtype([("isFixed", np.bool),
                   ("parameterize",np.bool),    #此权重是否需要参数化
                   ("initialValue", np.float64),   #此权重的重要性？可以在计算梯度时按照重要性对权重进行缩放。
                   ("a", np.float64),
                   ("b", np.float64)])

Variable = np.dtype([("isEvidence", np.int8),
                     ("initialValue", np.int64),
                     ("dataType", np.int16),
                     ("cardinality", np.int64),
                     ("vtf_offset", np.int64)])

Factor = np.dtype([("factorFunction", np.int16),
                   ("weightId", np.int64),
                   ("featureValue", np.float64),
                   ("arity", np.int64),           #与因子相连的变量数目
                   ("ftv_offset", np.int64)])     #在FactorToVar中的偏移量


FactorToVar = np.dtype([("vid", np.int64),
                        ("x", np.float64),
                        ("theta", np.float64),
                        ("dense_equal_to", np.int64)])

VarToFactor = np.dtype([("value", np.int64),
                        ("factor_index_offset", np.int64),
                        ("factor_index_length", np.int64)])

UnaryFactorOpt = np.dtype([('vid', np.int64),
                           ('weightId', np.int64)])

##################### add for extend ######################
AlphaBound = np.dtype([("lowerBound", np.float64),
                    ("upperBound", np.float64)])
TauBound =  np.dtype([("lowerBound", np.float64),
                    ("upperBound", np.float64)])

SampleList = np.dtype([("vid", np.int64)])


#记录每个weight拥有的factor在FactorToWeight中的偏移位置和长度
WeightToFactor = np.dtype([("weightId", np.int64),
                           ("weight_index_offset", np.int64),
                           ("weight_index_length", np.int64)])
FactorToWeight = np.dtype([("factorId", np.int64)])