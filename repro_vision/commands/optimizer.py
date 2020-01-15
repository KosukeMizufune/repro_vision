from repro_vision import optimizers


def get_optimizer(net, config):
    optimizer = getattr(optimizers, config['opt_name'])
    return optimizer(net.parameters(), **config['params'])
