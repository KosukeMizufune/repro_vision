from repro_vision import models
from repro_vision.functions import loss


def get_net(config):
    return getattr(models, config['model_name'])(**config['params'])


def get_criterion(config):
    module = getattr(loss, config['loss_name'])
    if config.get('params'):
        return module(**config["params"])
    else:
        return module()
