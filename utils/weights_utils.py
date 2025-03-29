from collections import OrderedDict


def weights_substraction(weights0, weights1):
    # substraction for for state_dicts
    ret = OrderedDict()
    for k in weights0:
        ret[k] = weights0[k] - weights1[k]
    return ret


def norm_l2(weights):
    ret = 0
    for k, v in weights.items():
        value = v.data.norm(2)
        ret += value.item() ** 2
    return ret ** 0.5


def norm_linf(weights):
    ret = 0
    for k, v in weights.items():
        ret = max(ret, v.data.abs().max().item())
    return ret


def norm(weights):
    return norm_l2(weights)
