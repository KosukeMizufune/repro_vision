from logging import getLogger
from pathlib import Path

import click
from ruamel import yaml
from ignite.engine import create_supervised_trainer, Events
from ignite.metrics import Average

from repro_vision.commands.dataset import (get_loaders, get_transforms,
                                           get_labels)
from repro_vision.commands.optimizer import get_optimizer
from repro_vision.commands.model import get_model
from repro_vision.commands.trainer import (prepare_batch, create_supervised_evaluator,  # noqa
                                           get_metrics, TrainExtension)

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
    net, criterion = get_model(n_class=len(label_names), **config['model'])
    optimizer = get_optimizer(net, **config['optimizer'])

    trainer = create_supervised_trainer(net, optimizer, criterion, device,
                                        prepare_batch=prepare_batch)
    metric_loss = Average()
    metric_loss.attach(trainer, 'loss')
    metrics = get_metrics(label_names, config['evaluate'])
    metric_names = list(metrics.keys())
    evaluator = create_supervised_evaluator(net, metrics, device,
                                            prepare_batch=prepare_batch)

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
