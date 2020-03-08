import logging

from repro_vision import models


def get_model(model_name, net_name, net_params, loss_params, n_class,
              logger=None, **kwargs):
    logger = logger or logging.getLogger(__name__)
    module = getattr(models, model_name)
    return module.get_model(net_name, net_params, loss_params, n_class,
                            logger=logger, **kwargs)
