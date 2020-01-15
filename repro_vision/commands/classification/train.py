import click
from ruamel import yaml

from repro_vision.commands.dataset import get_loaders, get_transforms
from repro_vision.commands.optimizer import get_optimizer
from repro_vision.commands.model import get_net, get_criterion
from repro_vision.commands.trainer import get_trainer, get_evaluator, train


@click.command()
@click.argument('config_file', type=click.Path(exists=True))
@click.option('--output_dir', type=click.Path(exists=False), default='results')
@click.option('--debug', is_flag=True)
@click.option('--device', default=None)
@click.option('--num_workers', type=int, default=0)
@click.pass_context
def main(ctx, config_file, **kwargs):
    with open(config_file) as stream:
        config = yaml.safe_load(stream)
    config.update(kwargs)
    config.update({'config_file': config_file})

    train_transforms = get_transforms(config['train_augment'])
    val_transforms = get_transforms(config['val_augment'])
    loaders = get_loaders(config['dataset'], train_transforms,
                          val_transforms, config['num_workers'])

    n_class = len(loaders[0].dataset.labels)
    config['model'].update({'params': {'n_classes': n_class}})
    net = get_net(config['model'])

    criterion = get_criterion(config['loss'])
    optimizer = get_optimizer(net, config['optimizer'])
    trainer = get_trainer(net, optimizer, criterion, config)
    evaluator = get_evaluator(net, config)

    train(trainer, evaluator, loaders, net, config)


if __name__ == "__main__":
    main()
