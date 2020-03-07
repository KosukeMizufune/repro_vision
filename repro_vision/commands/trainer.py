from datetime import datetime
import shutil
from pathlib import Path

from ignite.contrib.handlers import (ProgressBar, global_step_from_engine,
                                     TensorboardLogger, LRScheduler,
                                     create_lr_scheduler_with_warmup)
from ignite.contrib.handlers.tensorboard_logger import OutputHandler
from ignite.handlers import ModelCheckpoint
from ignite.engine import Events
import torch
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
import pandas as pd
import matplotlib.pyplot as plt

from repro_vision import trainers, evaluations


def get_trainer(net, optimizer, criterion, device, task):
    module = getattr(trainers, f'create_{task}_trainer')
    trainer = module(net, optimizer, criterion, device)
    return trainer


def get_metrics(eval_config):
    metrics = dict()
    for name, params in eval_config.items():
        if params:
            metrics[name] = getattr(evaluations, name)(**params)
        else:
            metrics[name] = getattr(evaluations, name)()
    return metrics


def get_evaluator(net, metrics, device, task):
    module = getattr(trainers, f'create_{task}_evaluator')
    evaluator = module(net, metrics, device)
    return evaluator


class TrainExtension:
    """Trainer Extension class.
    You can add tranier-extension method (e.g. Registration for TensorBoard,
    Learning Rate Scheduler, etc.) to this class. If you add, then you must
    add such method in also train.py.
    Args:
        res_root_dir (str): result root directory. outputs will be saved in
            ``{res_root_dir}/{task}/{timestamp}/`` directory.
    """
    def __init__(self, trainer, evaluator, res_dir='results', **kwargs):
        self.trainer = trainer
        self.evaluator = evaluator
        self.step_func = global_step_from_engine(trainer)
        self.start_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.res_dir = res_dir / self.start_datetime
        self.prefix = f"{self.start_datetime}"
        self.res_dir.mkdir(parents=True)

    def copy_configs(self, config_file):
        shutil.copy(config_file, self.res_dir / Path(config_file).name)

    def set_progressbar(self):
        """Attach ProgressBar.
        Args:
            trainer (ignite.Engine): trainer
            val_evaluator (ignite.Engine): validation evaluator.
            val_evaluator (ignite.Engine): test evaluator.
        """
        pbar = ProgressBar(persist=True)
        pbar.attach(self.trainer)
        pbar_val = ProgressBar(persist=True)
        pbar_val.attach(self.evaluator)

    def set_tensorboard(self, metrics):
        """Extension method for logging on tensorboard.
        Args:
            trainer (ignite.Engine): trainer
            val_evaluator (ignite.Engine): validation evaluator.
            val_evaluator (ignite.Engine): test evaluator.
        """
        logger = TensorboardLogger(
            log_dir=self.res_dir / 'tensorboard' / 'train'
        )
        _log_tensorboard(logger, self.trainer, f"{self.prefix}/train",
                         self.step_func, ["loss"])
        _log_tensorboard(logger, self.evaluator, f"{self.prefix}/val",
                         self.step_func, metrics)

    def save_model(self, model, save_interval=None, n_saved=1):
        """Extension method for saving model.
        This method saves model as a PyTorch model filetype (.pth). Saved
        file will be saved on `self.res_dir / model / {model_class_name}.pth`.
        Args:
            trainer (ignite.Engine): trainer
            model (torch.nn.Module): model class.
            save_interval (int): Number of epoch interval in which model should
                be kept on disk.
            n_saved (int): Number of objects that should be kept on disk. Older
                files will be removed. If set to None, all objects are kept.
        """
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        save_handler = ModelCheckpoint(
            self.res_dir / 'model',
            model.__class__.__name__,
            save_interval=save_interval,
            n_saved=n_saved
        )
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED, save_handler,
                                       {'epoch': model})

    def show_config_on_tensorboard(self, config):
        config_table = pd.json_normalize(config).T
        config_table.index = config_table.index.str.split('.', expand=True)
        config_table = config_table.reset_index().fillna('')

        fig = plt.figure(figsize=(6, 6), dpi=200)
        ax = fig.add_subplot(111)
        n_col = len(config_table.columns)
        ax.table(cellText=config_table.values, loc='center', cellLoc='center',
                 colLabels=[''] * (n_col - 1) + ['Parameter'],
                 colColours=['lightgray'] * n_col,
                 bbox=[0, 0, 1, 1])
        ax.axis('off')

        writer = SummaryWriter(log_dir=self.res_dir / 'tensorboard' / 'config')
        writer.add_figure(tag=f"{self.prefix}/config", figure=fig)
        writer.close()

    def print_metrics(self, metrics):
        """Extension method for printing metrics.
        For now, this method prints only validation AP@0.5, mAP@0.5, and
        traning loss.
        Args:
            trainer (ignite.Engine): trainer
            val_evaluator (ignite.Engine): validation evaluator.
        """
        @self.trainer.on(Events.EPOCH_COMPLETED)
        def compute_metrics(engine):
            val_metrics = self.evaluator.state.metrics
            print(f"Train loss is {self.trainer.state.metrics['loss']}")
            for metric in metrics:
                print(f"Val {metric} is {val_metrics.get(metric)}")

    def schedule_lr(self, optimizer, name, params, warmup_start=None,
                    warmup_end=None, warmup_duration=None):
        if name is None:
            return None
        lr_scheduler = self._get_lr_scheduler(name)(optimizer, **params)
        if warmup_start and warmup_end and warmup_duration:
            scheduler = \
                create_lr_scheduler_with_warmup(lr_scheduler,
                                                warmup_start_value=warmup_start,
                                                warmup_end_value=warmup_end,
                                                warmup_duration=warmup_duration)
        else:
            scheduler = LRScheduler(lr_scheduler)
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED, scheduler)

    @staticmethod
    def _get_lr_scheduler(name):
        return getattr(lr_scheduler, name)


def _log_tensorboard(logger, engine, tag, global_step_transform, metric_names,
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
