from repro_vision import optimizers


def get_optimizer(net, name, params):
    optimizer = getattr(optimizers, name)
    return optimizer(net.parameters(), **params)
