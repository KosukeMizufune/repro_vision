import numpy as np
import torch


def resize_bbox(bbox, in_size, out_size):
    """Resize bounding boxes according to image resize.
    Args:
        bbox (~numpy.ndarray): See the table below.
        in_size (tuple): A tuple of length 2. The height and the width
            of the image before resized.
        out_size (tuple): A tuple of length 2. The height and the width
            of the image after resized.
    .. csv-table::
        :header: name, shape, dtype, format
        :obj:`bbox`, ":math:`(R, 4)`", :obj:`float32`, \
        ":math:`(y_{min}, x_{min}, y_{max}, x_{max})`"
    Returns:
        ~numpy.ndarray:
        Bounding boxes rescaled according to the given image shapes.
    """
    bbox = bbox.copy()
    y_scale = float(out_size[0]) / in_size[0]
    x_scale = float(out_size[1]) / in_size[1]
    bbox[:, 0] = y_scale * bbox[:, 0]
    bbox[:, 2] = y_scale * bbox[:, 2]
    bbox[:, 1] = x_scale * bbox[:, 1]
    bbox[:, 3] = x_scale * bbox[:, 3]
    return bbox


def _bbox_iou_numpy(bbox_a, bbox_b):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.
    IoU is calculated as a ratio of area of the intersection
    and area of the union.
    This function accepts both :obj:`numpy.ndarray` and :obj:`cupy.ndarray` as
    inputs. Please note that both :obj:`bbox_a` and :obj:`bbox_b` need to be
    same type.
    The output is same type as the type of the inputs.
    Args:
        bbox_a (np.ndarray): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        bbox_b (np.ndarray): An array similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.
    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
        box in :obj:`bbox_b`.
    TODO: This function will be removed.
        We'll port all numpy method to torch.Tensor.
    """
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        raise IndexError

    # top left
    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    # bottom right
    br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])

    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    # Shape of area_a is (N), but shape of area_i is (N, K).
    # You must broadcast area_a.
    return area_i / (area_a[:, None] + area_b - area_i)


def _bbox_iou_tensor(bbox_a, bbox_b):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.
    IoU is calculated as a ratio of area of the intersection
    and area of the union.
    This function accepts :obj:`torch.Tensor` as
    inputs. Please note that both :obj:`bbox_a` and :obj:`bbox_b` need to be
    same type.
    The output is same type as the type of the inputs.
    Args:
        bbox_a (torch.Tensor): A tensor whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`torch.float32`.
        bbox_b (torch.Tensor): A tensor similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`torch.float32`.
    Returns:
        torch.Tensor:
        A tensor whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
        box in :obj:`bbox_b`.
    """
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        raise IndexError

    # top left
    tl = torch.max(bbox_a[:, :2].unsqueeze(1), bbox_b[:, :2])
    # bottom right
    br = torch.min(bbox_a[:, 2:].unsqueeze(1), bbox_b[:, 2:])

    area_i = torch.prod(br - tl, dim=2) * (tl < br).all(dim=2)
    area_a = torch.prod(bbox_a[:, 2:] - bbox_a[:, :2], dim=1)
    area_b = torch.prod(bbox_b[:, 2:] - bbox_b[:, :2], dim=1)
    # Shape of area_a is (N), but shape of area_i is (N, K).
    # You must broadcast area_a.
    return area_i / (area_a.unsqueeze(1) + area_b - area_i)


def bbox_iou(bbox_a, bbox_b):
    if isinstance(bbox_a, np.ndarray) and isinstance(bbox_b, np.ndarray):
        return _bbox_iou_numpy(bbox_a, bbox_b)
    elif isinstance(bbox_a, torch.Tensor) and isinstance(bbox_b, torch.Tensor):
        return _bbox_iou_tensor(bbox_a, bbox_b)
