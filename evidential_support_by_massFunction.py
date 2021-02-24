from copy import copy
import gml_utils
from pyds import MassFunction

def construct_mass_function_for_propensity(uncertain_degree, label_prob, unlabel_prob):
    '''
    # l: support for labeling
    # u: support for unalbeling
    '''
    return MassFunction({'l': (1 - uncertain_degree) * label_prob,
                         'u': (1 - uncertain_degree) * unlabel_prob,
                         'lu': uncertain_degree})


def labeling_propensity_with_ds(mass_functions):
    combined_mass = gml_utils.combine_evidences_with_ds(mass_functions, normalization=True)
    return combined_mass