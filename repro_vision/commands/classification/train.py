from logging import getLogger
from pathlib import Path

import click
from ruamel import yaml
from ignite.engine import Events

from repro_vision.commands.dataset import (get_loaders, get_transforms,
                                           get_labels)
from repro_vision.commands.optimizer import get_optimizer
from repro_vision.commands.model import get_net, get_loss
from repro_vision.commands.trainer import (get_trainer, get_evaluator,
                                           get_metrics, TrainExtension)

TASK = 'classification'

logger = getLogger(__name__)


@click.command()
@click.argument('config_file', type=click.Path(exists=True))
@click.option('--dataset_root', type=click.Path(exists=True), default="data")
@click.option('--res_root_dir', type=click.Path(exists=False),
              default='results')
@click.option('--debug', is_flag=True)
@click.option('--device', default=None)
@click.option('--num_workers', type=int, default=0)
@click.pass_context
def main(ctx, config_file, dataset_root, res_root_dir, debug, device,
         num_workers, **kwargs):
    with open(config_file) as stream:
        config = yaml.safe_load(stream)

    train_transforms = get_transforms(config['train_augment'])
    val_transforms = get_transforms(config['val_augment'])
    train_loader, val_loader = get_loaders(train_transforms=train_transforms,
                                           val_transforms=val_transforms,
                                           dataset_root=dataset_root,
                                           num_workers=num_workers,
                                           **config['dataset'])
    label_names = get_labels(train_loader)
    net = get_net(n_class=len(label_names), **config['model'], logger=logger)
    criterion = get_loss(**config['loss'], logger=logger)
    optimizer = get_optimizer(net, **config['optimizer'])

    trainer = get_trainer(net, optimizer, criterion, device, TASK)
    metrics = get_metrics(config['evaluate'])
    metric_names = list(metrics.keys())
    evaluator = get_evaluator(net, metrics, device, TASK)

    @trainer.on(Events.EPOCH_COMPLETED)
    def compute_metrics(engine):
        evaluator.run(val_loader)

    res_dir = Path(res_root_dir) / config['dataset']['dataset_name']
    train_extend = TrainExtension(trainer, evaluator, res_dir)
    train_extend.print_metrics(metric_names)
    train_extend.set_progressbar()
    train_extend.schedule_lr(optimizer, **config['lr_schedule'])
    if not debug:
        train_extend.copy_configs(config_file)
        train_extend.set_tensorboard(metric_names)
        train_extend.save_model(net, **config['model_checkpoint'])
        train_extend.show_config_on_tensorboard(config)

    trainer.run(train_loader, max_epochs=config['epochs'])


if __name__ == "__main__":
    main()
