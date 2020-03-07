import logging

from repro_vision import models
from repro_vision.functions import loss


def get_net(n_class, name, params={}, logger=None):
    logger = logger or logging.getLogger(__name__)
    return getattr(models, name)(n_class=n_class, logger=logger, **params)


def get_loss(name, params={}, logger=None):
    logger = logger or logging.getLogger(__name__)
    return getattr(loss, name)(logger=logger, **params)
