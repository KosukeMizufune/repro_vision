import functools

import torch


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


def load_pth(model, params, correspondence=None):
    """Load PyTorch weight file `*.pth`.
    Since parameter names in weight file are sometimes different from model
    parameter names, correspondence argument is used for its compatibility.
    If some parameter names don't exist in model or parameter shape doesn't
    match, such parameters will be ignored.
    Args:
        model (torch.nn.Module): Model.
        params (dict of torch.Tensor): Weight parameters.
        correspondence (dict): Correspondence dict of parameters between
            weight file and model. Its key are parameter names in weight file,
            and its values are those in model.
    """
    for name, param in params.items():
        if correspondence.get(name) and \
                rgetattr(model, correspondence[name]).shape == param.shape:
            rsetattr(model, correspondence[name], torch.nn.Parameter(param))
