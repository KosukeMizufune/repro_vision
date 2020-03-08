from datetime import datetime
import shutil
from pathlib import Path
import collections.abc as collections

from ignite.contrib.handlers import (ProgressBar, global_step_from_engine,
                                     TensorboardLogger, LRScheduler,
                                     create_lr_scheduler_with_warmup)
from ignite.contrib.handlers.tensorboard_logger import OutputHandler
from ignite.handlers import ModelCheckpoint
from ignite.engine import Events, Engine
import torch
from torch._six import string_classes
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from repro_vision import evaluations


def convert_tensor(input_, device=None, non_blocking=False):
    """Move tensors to relevant device."""
    def _func(tensor):
        return tensor.to(device=device, non_blocking=non_blocking) if device else tensor

    return apply_to_tensor(input_, _func)


def apply_to_tensor(input_, func):
    """Apply a function on a tensor or mapping, or sequence of tensors.
    """
    return apply_to_type(input_, torch.Tensor, func)


def apply_to_type(input_, input_type, func):
    """Apply a function on a object of `input_type` or mapping, or sequence of objects of `input_type`.
    """
    if isinstance(input_, input_type):
        return func(input_)
    elif isinstance(input_, string_classes) or isinstance(input_, np.ndarray):
        return input_
    elif isinstance(input_, collections.Mapping):
        return {k: apply_to_type(sample, input_type, func) for k, sample in input_.items()}
    elif isinstance(input_, collections.Sequence):
        return [apply_to_type(sample, input_type, func) for sample in input_]
    else:
        raise TypeError(("input must contain {}, dicts or lists; found {}"
                         .format(input_type, type(input_))))


def to_onehot(indices, num_classes):
    """Convert a tensor of indices of any shape `(N, ...)` to a
    tensor of one-hot indicators of shape `(N, num_classes, ...) and of type uint8. Output's device is equal to the
    input's device`.
    """
    onehot = torch.zeros(indices.shape[0], num_classes, *indices.shape[1:],
                         dtype=torch.uint8,
                         device=indices.device)
    return onehot.scatter_(1, indices.unsqueeze(1), 1)


def prepare_batch(batch, device=None, non_blocking=False):
    """Prepare batch for training: pass to a device with options.
    """
    x, *labels = batch
    x = convert_tensor(x, device=device, non_blocking=non_blocking)
    if len(labels) == 1:
        y = convert_tensor(labels[0], device=device, non_blocking=non_blocking)
    else:
        y = [convert_tensor(label, device=device, non_blocking=non_blocking)
             for label in labels]
    return (x, y)


def get_metrics(labels, eval_config):
    metrics = dict()
    for name, params in eval_config.items():
        if params:
            metrics[name] = getattr(evaluations, name)(labels=labels, **params)
        else:
            metrics[name] = getattr(evaluations, name)(labels=labels)
    return metrics


def create_supervised_evaluator(model, metrics=None, device=None,
                                non_blocking=False,
                                prepare_batch=prepare_batch):
    """
    Factory function for creating an evaluator for supervised models.
    Args:
        model (`torch.nn.Module`): the model to train.
        metrics (dict of str - :class:`~ignite.metrics.Metric`): a map of
            metric names to Metrics.
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
        non_blocking (bool, optional): if True and this copy is between CPU and
            GPU, the copy may occur asynchronously with respect to the host.
            For other cases, this argument has no effect.
        prepare_batch (callable, optional): function that receives `batch`,
            `device`, `non_blocking` and outputs tuple of tensors
            `(batch_x, batch_y)`.
        output_transform (callable, optional): function that receives 'x', 'y',
            'y_pred' and returns value to be assigned to engine's state.output
            after each iteration. Default is returning `(y_pred, y,)` which
            fits output expected by metrics. If you change it you should use
            `output_transform` in metrics.
    Note:
        `engine.state.output` for this engine is defind by `output_transform`
        parameter and is a tuple of `(batch_pred, batch_y)` by default.
    Returns:
        Engine: an evaluator engine with supervised inference function.
    """
    metrics = metrics or {}

    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    if device:
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            img, targets = prepare_batch(
                batch, device=device, non_blocking=non_blocking
            )
            preds = model.predict(img)
        return (preds, targets)

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)
    return engine


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
