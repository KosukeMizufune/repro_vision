import logging

from repro_vision import models
from repro_vision.functions import loss


def get_net(name, params={}, logger=None):
    logger = logger or logging.getLogger(__name__)
    return getattr(models, name)(logger=logger, **params)


def get_loss(name, params={}, logger=None):
    logger = logger or logging.getLogger(__name__)
    return getattr(loss, name)(logger=logger, **params)
