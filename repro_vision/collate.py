import re

import torch
from torch._six import container_abcs, string_classes, int_classes
import numpy as np


np_str_obj_array_pattern = re.compile(r'[SaUO]')
collate_classification_err_msg_format = (
    "collate_classification: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")


def collate_classification(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)

    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(collate_classification_err_msg_format.format(elem.dtype))

            return collate_classification([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: collate_classification([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(collate_classification(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        transposed = zip(*batch)
        return [collate_classification(samples) for samples in transposed]

    raise TypeError(collate_classification_err_msg_format.format(elem_type))


def _parse_tensor_or_numpy(x):
    if isinstance(x, torch.Tensor):
        return x
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x.transpose(2, 0, 1).astype(np.float32))
    else:
        raise TypeError("Image should be tensor or np.ndarray,"
                        f"but got {type(x)}.")


def collate_detection_train(batch):
    images = []
    rest = []
    for i, elems in enumerate(zip(*batch)):
        if i == 0:
            images = torch.stack(elems)
        else:
            rest.append(elems)
    return (images, *tuple(rest))


def collate_detection_val(batch):
    """Collate function for object detection.
    This function tensorizes only image. ``torch.utils.data.Dataloader`` will
    use this function as an argument of ``collate_fn``.
    Args:
        batch (list): List of samples.
    Returns:
        tuple: Concated batch where each is list of elements.
    """
    images = []
    rest = []
    for i, elems in enumerate(zip(*batch)):
        if i == 0:
            images = [_parse_tensor_or_numpy(elem) for elem in elems]
        else:
            rest.append(elems)

    return (images, *tuple(rest))
