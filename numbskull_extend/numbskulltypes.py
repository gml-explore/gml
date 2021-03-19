"""TODO."""

from __future__ import print_function, absolute_import
import numpy as np

Meta = np.dtype([('weights', np.int64),
                 ('variables', np.int64),
                 ('factors', np.int64),
                 ('edges', np.int64)])


Weight = np.dtype([("isFixed", np.bool),        #Whether the weight needs to be learned, it needs to be false, otherwise it is true
                   ("parameterize",np.bool),    #Does this weight need to be parameterized, 0,1
                   ("initialValue", np.float64),   #How important is this weight? The weight can be scaled according to importance when calculating the gradient.
                   ("a", np.float64),    #tau
                   ("b", np.float64)])   #alpha

Variable = np.dtype([("isEvidence", np.int8),   #if vidence variable
                     ("initialValue", np.int64), #Variable initial value
                     ("dataType", np.int16),    #datatype=0 means it is a bool variable, when it is 1, it means a non-Boolean variable.
                     ("cardinality", np.int64), #The number of variable values
                     ("vtf_offset", np.int64)])

Factor = np.dtype([("factorFunction", np.int16), #Factor function type
                   ("weightId", np.int64),
                   ("featureValue", np.float64),
                   ("arity", np.int64),           #Number of variables connected to the factor
                   ("ftv_offset", np.int64)])     #Offset in FactorToVar


FactorToVar = np.dtype([("vid", np.int64),
                        ("x", np.float64),   # store faetureValue
                        ("theta", np.float64), #store theta
                        ("dense_equal_to", np.int64)])

VarToFactor = np.dtype([("value", np.int64),
                        ("factor_index_offset", np.int64),
                        ("factor_index_length", np.int64)])

UnaryFactorOpt = np.dtype([('vid', np.int64),
                           ('weightId', np.int64)])

##################### add for extend ######################
AlphaBound = np.dtype([("lowerBound", np.float64),   #Alpha lowerBound
                    ("upperBound", np.float64)])  #Alpha upperBound
TauBound =  np.dtype([("lowerBound", np.float64),  #tau lowerBound
                    ("upperBound", np.float64)]) #tau upperBound

SampleList = np.dtype([("vid", np.int64)])   #Store the variable ID that is out of order during balancing


#Record the offset position and length of the factor owned by each weight in FactorToWeight
WeightToFactor = np.dtype([("weightId", np.int64),
                           ("weight_index_offset", np.int64),
                           ("weight_index_length", np.int64)])
FactorToWeight = np.dtype([("factorId", np.int64)])