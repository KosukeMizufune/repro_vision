from datetime import datetime
from pathlib import Path
import shutil

from ignite.contrib.handlers import (ProgressBar, global_step_from_engine,
                                     tensorboard_logger)
from ignite.contrib.handlers.tensorboard_logger import OutputHandler
from ignite.handlers import ModelCheckpoint
from ignite.engine import Events

from repro_vision import trainers, evaluations


def get_trainer(net, optimizer, criterion, config):
    module = getattr(trainers, f'create_{config["task"]}_trainer')
    trainer = module(net, optimizer, criterion, config['device'])
    return trainer


def get_metrics(config):
    metrics = dict()
    for name, params in config.items():
        if params:
            metrics[name] = getattr(evaluations, name)(**params)
        else:
            metrics[name] = getattr(evaluations, name)()
    return metrics


def get_evaluator(net, config):
    metrics = get_metrics(config['metrics'])
    module = getattr(trainers, f'create_{config["task"]}_evaluator')
    evaluator = module(net, metrics, config['device'])
    return evaluator


def log_tensorboard(logger, engine, tag, global_step_transform, metric_names,
                    event_name=Events.EPOCH_COMPLETED):
    logger.attach(
        engine,
        log_handler=OutputHandler(
            tag=tag,
            metric_names=metric_names,
            global_step_transform=global_step_transform
        ),
        event_name=event_name
    )


def train(trainer, evaluator, loaders, net, config):
    pbar = ProgressBar(persist=True)
    pbar.attach(trainer)
    pbar.attach(evaluator)

    config_file = config['config_file']
    train_loader, val_loader = loaders

    @trainer.on(Events.EPOCH_COMPLETED)
    def compute_metrics(engine):
        evaluator.run(val_loader)
        print(f"Training loss is {trainer.state.metrics['loss']}")

    if config['debug']:
        tb_logger = None
    else:
        start_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        res_dir = Path(config['output_dir']) / start_datetime

        # Copy config
        res_dir.mkdir(parents=True)
        shutil.copy(config_file, res_dir / Path(config_file).name)

        step_func = global_step_from_engine(trainer)

        # TensorBoard
        tb_logger = tensorboard_logger.TensorboardLogger(
            log_dir=res_dir / 'tensorboard'
        )
        log_tensorboard(tb_logger, trainer,
                        f"{start_datetime}/training",
                        step_func, ['loss'])
        log_tensorboard(tb_logger, evaluator,
                        f"{start_datetime}/validation",
                        step_func, config['metrics'])

        # Save Model
        save_handler = ModelCheckpoint(
            res_dir / 'model',
            start_datetime,
            **config['model_checkpoint']
        )
        trainer.add_event_handler(Events.EPOCH_COMPLETED, save_handler,
                                  {'epoch': net})
    trainer.run(train_loader, max_epochs=config['epochs'])
    if tb_logger:
        tb_logger.close()
