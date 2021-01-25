"""TODO."""

from __future__ import print_function, absolute_import
import numba
from numba import jit
import numpy as np
import math
import random
from numbskull_extend.inference import draw_sample, eval_factor


@jit(cache=True, nogil=True)
def learnthread(shardID, nshards, step, regularization, reg_param,
                truncation,var_copy, weight_copy, weight,
                variable, factor, fmap,
                vmap, factor_index, Z, fids,
                var_value, var_value_evid,
                weight_value, learn_non_evidence,
                poential_weight,alpha_bound,tau_bound,sample_list=None,wmap=None,wfactor=None):
    """TODO."""
    # Identify start and end variable
    nvar = variable.shape[0]
    start = (shardID * nvar) // nshards
    end = ((shardID + 1) * nvar) // nshards
    if sample_list == None:     #sample_list为None表示不需要平衡化
        for var_samp in range(start, end):
            if variable[var_samp]["isEvidence"] == 4:
                # This variable is not owned by this machine
                continue
            sample_and_sgd(var_samp, step, regularization, reg_param, truncation,
                           var_copy, weight_copy, weight, variable,
                           factor, fmap, vmap,
                           factor_index, Z[shardID], fids[shardID], var_value,
                           var_value_evid, weight_value, learn_non_evidence,
                           poential_weight,alpha_bound,tau_bound)
    else:      #需要平衡化
        sample_num = sample_list.shape[0]
        start = (shardID * sample_num) // nshards
        end = ((shardID + 1) * sample_num) // nshards
        sample_num = sample_list.shape[0]
        for i in range(0,sample_num) :
            var_samp = sample_list[i]['vid']
            if variable[var_samp]["isEvidence"] == 4:
                # This variable is not owned by this machine
                continue
            sample_and_sgd(var_samp, step, regularization, reg_param, truncation,
                           var_copy, weight_copy, weight, variable,
                           factor, fmap, vmap,factor_index, Z[shardID],
                           fids[shardID], var_value,var_value_evid,
                           weight_value, learn_non_evidence,
                           poential_weight,alpha_bound,tau_bound)


@jit(cache=True, nogil=True)
def learnthread_bgd(shardID, nshards, step, regularization, reg_param,
                truncation,var_copy, weight_copy, weight,
                variable, factor, fmap,
                vmap, factor_index, Z, fids,
                var_value, var_value_evid,
                weight_value, learn_non_evidence,
                poential_weight,alpha_bound,tau_bound,sample_list,wmap,wfactor):
    for wid in range(0,len(weight)):
        if weight[wid]["isFixed"]:
            continue
        else:
            sample_and_bgd(wid,step, regularization, reg_param, truncation,
                       var_copy, weight_copy, weight, variable,
                       factor, fmap,vmap, factor_index, Z[shardID],
                       fids[shardID], var_value, var_value_evid,
                       weight_value, learn_non_evidence,
                       alpha_bound,tau_bound,wmap,wfactor)


@jit(nopython=True, cache=True, nogil=True)
def sample_and_bgd(wid,step, regularization, reg_param, truncation,
                   var_copy, weight_copy, weight, variable,
                   factor, fmap,vmap, factor_index, Z,
                   fids, var_value, var_value_evid,
                   weight_value, learn_non_evidence,
                   alpha_bound,tau_bound,wmap,wfactor):    #批量梯度下降不需要考虑poential_weight
    #1. 计算梯度和
    weight_id = wmap[wid]["weightId"]
    weight_index_offset = wmap[wid]["weight_index_offset"]
    weight_index_length = wmap[wid]["weight_index_length"]
    #找到此权重相关的每一个factor
    factor_count = weight_index_length   #此权重拥有的因子个数
    for fIndex in range(weight_index_offset,weight_index_offset+weight_index_length):
        factor_id = wfactor[fIndex]["factorId"]
        ftv_offset = factor[factor_id]["ftv_offset"]
        ftv_length = factor[factor_id]["arity"]

        var_count = ftv_length          #此因子拥有的变量个数
        gradient_sum = 0  #不需要参数化时，所有梯度的和
        gradient1_sum = 0 #需要参数化时，参数1的梯度和
        gradient2_sum = 0  # 需要参数化时，参数2的梯度和
        # 找到每一个factor相关的每一个变量
        for vIndex in range(ftv_offset,ftv_offset+ftv_length):
            var_samp = fmap[vIndex]["vid"]
            if variable[var_samp]["isEvidence"] != 1:
                evidence = draw_sample(var_samp, var_copy, weight_copy,
                                       weight, variable, factor,
                                       fmap, vmap, factor_index, Z,
                                       var_value_evid, weight_value)
                # If evidence then store the initial value in a tmp variable
            # then sample and compute the gradient.
            else:
                evidence = variable[var_samp]["initialValue"]
            var_value_evid[var_copy][var_samp] = evidence
            # Sample the variabl e
            proposal = draw_sample(var_samp, var_copy, weight_copy, weight,
                                   variable, factor, fmap, vmap,
                                   factor_index, Z, var_value, weight_value)
            var_value[var_copy][var_samp] = proposal
            if not learn_non_evidence and variable[var_samp]["isEvidence"] != 1:
                return
            truncate = random.random() < 1.0 / truncation if regularization == 1 else False
            p0 = eval_factor(factor_id, var_samp,
                             evidence, var_copy,
                             variable, factor, fmap,
                             var_value_evid)
            p1 = eval_factor(factor_id, var_samp,
                             proposal, var_copy,
                             variable, factor, fmap,
                             var_value)
            # if need parameterize
            if weight[factor[factor_id]['weightId']]['parameterize']:
                x = fmap[factor[factor_id]["ftv_offset"]]['x']
                theta = fmap[factor[factor_id]["ftv_offset"]]['theta']
                a = weight[factor[factor_id]['weightId']]['a']
                b = weight[factor[factor_id]['weightId']]['b']
                gradient1 = (p1 - p0) * theta * factor[factor_id]["featureValue"] * (x - b)
                gradient2 = (p1 - p0) * theta * factor[factor_id]["featureValue"] * (-a)
                gradient1_sum += gradient1
                gradient2_sum += gradient2
            # if not need parameterize
            else:
                gradient = (p1 - p0) * factor[factor_id]["featureValue"]
                gradient_sum += gradient
        #求平均值
        # print("权重开始平均")
        gradient1_sum /= var_count
        gradient2_sum /= var_count
        gradient_sum /= var_count

    #2.更新参数，分为需要参数化的和不需要参数化的
    #if need parameterize
    if weight[factor[factor_id]['weightId']]['parameterize']:
            if regularization == 2:  # 是否需要正则化
                a *= (1.0 / (1.0 + reg_param * step))
                a -= step * gradient1_sum/factor_count
                b *= (1.0 / (1.0 + reg_param * step))
                b -= step * gradient2_sum/factor_count
            elif regularization == 1:
                # Truncated Gradient
                # "Sparse Online Learning via Truncated Gradient"
                #  Langford et al. 2009
                a -= step * gradient1_sum/factor_count
                b -= step * gradient2_sum/factor_count
                if truncate:
                    l1delta = reg_param * step * truncation
                    a = max(0, a - l1delta) if a > 0 else min(0, a + l1delta)
                    b = max(0, b - l1delta) if b > 0 else min(0, b + l1delta)
            else:
                a -= step * gradient1_sum/factor_count
                b -= step * gradient2_sum/factor_count
            if a < tau_bound[factor[factor_id]['weightId']]['lowerBound']:
                a = tau_bound[factor[factor_id]['weightId']]['lowerBound']
            elif a > tau_bound[factor[factor_id]['weightId']]['upperBound']:
                a = tau_bound[factor[factor_id]['weightId']]['upperBound']
            if b > alpha_bound[factor[factor_id]['weightId']]['upperBound']:
                b = alpha_bound[factor[factor_id]['weightId']]['upperBound']
            elif b < alpha_bound[factor[factor_id]['weightId']]['lowerBound']:
                b = alpha_bound[factor[factor_id]['weightId']]['lowerBound']
            w = theta * a * (x - b)
            weight[factor[factor_id]['weightId']]['a'] = a
            weight[factor[factor_id]['weightId']]['b'] = b
    # if not need parameterize
    else:
        w = weight_value[weight_copy][weight_id]
        if regularization == 2:
            w *= (1.0 / (1.0 + reg_param * step))
            w -= step * gradient_sum /factor_count
        elif regularization == 1:
            # Truncated Gradient
            # "Sparse Online Learning via Truncated Gradient"
            #  Langford et al. 2009
            w -= step * gradient_sum /factor_count
            if truncate:
                l1delta = reg_param * step * truncation
                w = max(0, w - l1delta) if w > 0 else min(0, w + l1delta)
        else:
            w -= step * gradient_sum /factor_count
    # print("权重开始更新")
    weight_value[weight_copy][weight_id] = w
    weight[factor[factor_id]['weightId']]['initialValue'] = w


@jit(nopython=True, cache=True, nogil=True)
def get_factor_id_range(variable, vmap, var_samp, val):
    """TODO."""
    varval_off = val
    if variable[var_samp]["dataType"] == 0:
        varval_off = 0
    vtf = vmap[variable[var_samp]["vtf_offset"] + varval_off]
    start = vtf["factor_index_offset"]
    end = start + vtf["factor_index_length"]
    return (start, end)

@jit(nopython=True, cache=True, nogil=True)
def sample_and_sgd(var_samp, step, regularization, reg_param, truncation,
                   var_copy, weight_copy, weight, variable,
                   factor, fmap,vmap, factor_index, Z,
                   fids, var_value, var_value_evid,
                   weight_value, learn_non_evidence,
                   poential_weight,alpha_bound,tau_bound):
    """TODO."""
    # If learn_non_evidence sample twice.
    # The method corresponds to expectation-conjugate descent.
    if variable[var_samp]["isEvidence"] != 1:
        evidence = draw_sample(var_samp, var_copy, weight_copy,
                               weight, variable, factor,
                               fmap, vmap, factor_index, Z,
                               var_value_evid, weight_value)
        # If evidence then store the initial value in a tmp variable
    # then sample and compute the gradient.
    else:
        evidence = variable[var_samp]["initialValue"]

    var_value_evid[var_copy][var_samp] = evidence
    # Sample the variabl e
    proposal = draw_sample(var_samp, var_copy, weight_copy, weight,
                           variable, factor, fmap, vmap,
                           factor_index, Z, var_value, weight_value)

    var_value[var_copy][var_samp] = proposal
    if not learn_non_evidence and variable[var_samp]["isEvidence"] != 1:
        return
    # Compute the gradient and update the weights
    # Iterate over corresponding factors

    range_fids = get_factor_id_range(variable, vmap, var_samp, evidence)
    # TODO: is it possible to avoid copying around fids
    if evidence != proposal:
        range_prop = get_factor_id_range(variable, vmap, var_samp, proposal)
        s1 = range_fids[1] - range_fids[0]
        s2 = range_prop[1] - range_prop[0]
        s = s1 + s2
        fids[:s1] = factor_index[range_fids[0]:range_fids[1]]
        fids[s1:s] = factor_index[range_prop[0]:range_prop[1]]
        fids[:s].sort()
    else:
        s = range_fids[1] - range_fids[0]
        fids[:s] = factor_index[range_fids[0]:range_fids[1]]

    truncate = random.random() < 1.0 / truncation if regularization == 1 else False
    # go over all factor ids, ignoring dupes
    last_fid = -1  # numba 0.28 would complain if this were None
    for factor_id in fids[:s]:
        if factor_id == last_fid:
            continue
        last_fid = factor_id
        weight_id = factor[factor_id]["weightId"]
        if weight[weight_id]["isFixed"]:
            continue
        # Compute Gradient
        p0 = eval_factor(factor_id, var_samp,
                         evidence, var_copy,
                         variable, factor, fmap,
                         var_value_evid)
        p1 = eval_factor(factor_id, var_samp,
                         proposal, var_copy,
                         variable, factor, fmap,
                         var_value)
        #if need parameterize
        if weight[factor[factor_id]['weightId']]['parameterize']:
            x = fmap[factor[factor_id]["ftv_offset"]]['x']
            theta = fmap[factor[factor_id]["ftv_offset"]]['theta']
            a = weight[factor[factor_id]['weightId']]['a']
            b = weight[factor[factor_id]['weightId']]['b']
            gradient1 = (p1 - p0) * theta * factor[factor_id]["featureValue"] * (x - b)
            gradient2 = (p1 - p0) * theta * factor[factor_id]["featureValue"] * (-a)
            if regularization == 2:  # 是否需要正则化
                a *= (1.0 / (1.0 + reg_param * step))
                a -= step * gradient1
                b *= (1.0 / (1.0 + reg_param * step))
                b -= step * gradient2
            elif regularization == 1:
            # Truncated Gradient
            # "Sparse Online Learning via Truncated Gradient"
            #  Langford et al. 2009
                a -= step * gradient1
                b -= step * gradient2
                if truncate:
                    l1delta = reg_param * step * truncation
                    a = max(0, a - l1delta) if a > 0 else min(0, a + l1delta)
                    b = max(0, b - l1delta) if b > 0 else min(0, b + l1delta)
            else:
                a -= step * gradient1
                b -= step * gradient2

            # if alpha_bound != None and tau_bound != None:
            if a < tau_bound[factor[factor_id]['weightId']]['lowerBound']:
                a = tau_bound[factor[factor_id]['weightId']]['lowerBound']
            elif a > tau_bound[factor[factor_id]['weightId']]['upperBound']:
                a = tau_bound[factor[factor_id]['weightId']]['upperBound']
            if b > alpha_bound[factor[factor_id]['weightId']]['upperBound']:
                b = alpha_bound[factor[factor_id]['weightId']]['upperBound']
            elif  b < alpha_bound[factor[factor_id]['weightId']]['lowerBound']:
                b = alpha_bound[factor[factor_id]['weightId']]['lowerBound']
            w = theta * a * (x - b)
            weight[factor[factor_id]['weightId']]['a'] = a
            weight[factor[factor_id]['weightId']]['b'] = b
        #如果不需要参数化
        else:
            gradient = (p1 - p0) * factor[factor_id]["featureValue"]
        # Update weight
            w = weight_value[weight_copy][weight_id]
            if regularization == 2:
                w *= (1.0 / (1.0 + reg_param * step))
                w -= step * gradient
            elif regularization == 1:
            # Truncated Gradient
            # "Sparse Online Learning via Truncated Gradient"
            #  Langford et al. 2009
                w -= step * gradient
                if truncate:
                    l1delta = reg_param * step * truncation
                    w = max(0, w - l1delta) if w > 0 else min(0, w + l1delta)
            else:
                w -= step * gradient
        weight_value[weight_copy][weight_id] = w
        weight[factor[factor_id]['weightId']]['initialValue'] = w
        if variable[var_samp]["isEvidence"] != 1:
            poential_weight[factor[factor_id]['weightId']] = w


