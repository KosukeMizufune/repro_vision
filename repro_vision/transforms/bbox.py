import torch
import numpy as np
from PIL import Image

from repro_vision.utils.bbox_utils import bbox_iou, resize_bbox


random_state = np.random.RandomState(333)


class RandomHorizontalFlipWithBbox(object):
    """Transform class of Random Horizontal Flip.
    Args:
        p (int): Flip probability of range :math:`[0, 1]`.
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        img, boxes, *rest_label = data
        _, w, _ = img.shape
        if random_state.random() < self.p:
            img = img[:, ::-1, :]
            boxes = self.flip_coord(boxes, w)
        return (img, boxes, *rest_label)

    @staticmethod
    def flip_coord(bbox, w):
        """Horizontally flip coordinates.
        Args:
            bbox (numpy.ndarray):
                Coordinates of bounding box of shape ``(n, 4)`` where ``n`` is
                the number of bounding boxes. Each bounding box is organized by
                :math:`(y_{min}, x_{min}, y_{max}, x_{max})`.
            w (int): Width of image.
        Returns:
            numpy.ndarray: Flipped bounding boxes of shape ``(n, 4)``.
        """
        # Because coordinates are 0-based, you must subtract 1 from width.
        x_max = (w - 1) - bbox[:, 1]
        x_min = (w - 1) - bbox[:, 3]
        bbox[:, 1] = x_min
        bbox[:, 3] = x_max
        return bbox


class RandomCropWithBbox(object):
    """Transform class of Random Crop With Constraints.
    Args:
        min_scale (int): Minimum scale of crop.
        max_scale (int): Maximum scale of crop.
        max_aspect_ratio (int or float): Maximum aspect ratio which is
        ``height/width``.
        constraints (tuple):
            Tuple of ``(min_iou, max_iou)``.
            If ``min_iou`` is ``None``, ``min_iou`` is 0, and if ``max_iou`` is
            ``None``, ``max_iou`` is 1.
            This argument controls threshold of iou.
        max_trial (int): The number of crop trial.
    """
    def __init__(self, min_scale=0.3, max_scale=1,
                 max_aspect_ratio=2, constraints=None,
                 max_trial=50):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.max_aspect_ratio = max_aspect_ratio
        self.max_trial = max_trial
        self.constraints = constraints
        if self.constraints is None:
            self.constraints = (
                (0.1, None),
                (0.3, None),
                (0.5, None),
                (0.7, None),
                (0.9, None),
                (None, 1),
            )

    def __call__(self, data):
        img, boxes, *rest_label = data
        img, param = self._random_crop_with_bbox_constraints(img, boxes)
        boxes, param = self._crop_bbox(boxes,
                                       y_slice=param['y_slice'],
                                       x_slice=param['x_slice'],
                                       allow_outside_center=False)
        if len(rest_label) > 1:
            rests = []
            for label in rest_label:
                rests.append(label[param['index']])
            return (img, boxes, *rests)
        else:
            return img, boxes, rest_label[0][param['index']]

    def _random_crop_with_bbox_constraints(self, img, bbox):
        """Randomly crop img with constaraints.
        Args:
            img (numpy.ndarray): Array of shape ``(h, w, c)``.
            bbox (numpy.ndarray): Array of shape ``(n, 2)``.
                Each bounding box is organized by
                :math:`(y_{min}, x_{min}, y_{max}, x_{max})`.
        Returns:
            tuple:
                Croped image of shape ``(h, w, c)`` and crop parameter
                ``param``. ``param`` is a dictionary of intermediate parameters
                whose contents are listed below with key, value-type and
                the description of the value.
            * **constraint** (*tuple*): The chosen constraint.
            * **y_slice** (*slice*): A slice in vertical direction used to \
                crop the input image.
            * **x_slice** (*slice*): A slice in horizontal direction used to \
                crop the input image.
        """
        h, w, _ = img.shape
        params = [{'constraint': None,
                   'y_slice': slice(0, h),
                   'x_slice': slice(0, w)}]

        if len(bbox) == 0:
            return img, params[0]

        for min_iou, max_iou in self.constraints:
            if min_iou is None:
                min_iou = 0
            if max_iou is None:
                max_iou = 1

            for _ in range(self.max_trial):
                scale = random_state.uniform(self.min_scale, self.max_scale)
                aspect_ratio = random_state.uniform(
                    max(1 / self.max_aspect_ratio, scale * scale),
                    min(self.max_aspect_ratio, 1 / (scale * scale)))

                crop_h = int(h * scale / np.sqrt(aspect_ratio))
                crop_w = int(w * scale * np.sqrt(aspect_ratio))

                crop_t = random_state.randint(h - crop_h)
                crop_l = random_state.randint(w - crop_w)
                crop_bb = np.array((
                    crop_t, crop_l, crop_t + crop_h, crop_l + crop_w))

                iou = bbox_iou(bbox, crop_bb[np.newaxis])
                if min_iou <= iou.min() and iou.max() <= max_iou:
                    params.append({
                        'constraint': (min_iou, max_iou),
                        'y_slice': slice(crop_t, crop_t + crop_h),
                        'x_slice': slice(crop_l, crop_l + crop_w)})
                    break

        param = random_state.choice(params)
        img = img[param['y_slice'], param['x_slice'], :]

        return img, param

    def _crop_bbox(self, bbox, y_slice=None, x_slice=None,
                   allow_outside_center=True):
        """Translate bounding boxes to fit within the cropped area of an image.
        Args:
            bbox (numpy.ndarray): Array of shape ``(n, 2)``.
                Each bounding box is organized by
                :math:`(y_{min}, x_{min}, y_{max}, x_{max})`.
            y_slice (slice): The slice of y axis.
            x_slice (slice): The slice of x axis.
            allow_outside_center (bool): If this argument is :obj:`False`,
                bounding boxes whose centers are outside of the cropped area
                are removed. The default value is :obj:`True`.
        Returns:
            tuple: Croped bounding boxes and params of dict whose contents are
                listed below with key, value-type and the description
                of the value.
            * **index** (*numpy.ndarray*): An array holding indices of used \
                bounding boxes.
            * **trancated_index** (*numpy.ndarray*): An array holding indices \
                of truncated bounding boxes, with respect to **returned** \
                :obj:`bbox`, rather than original :obj:`bbox`.
        """
        t, b = self._slice_to_bounds(y_slice)
        l, r = self._slice_to_bounds(x_slice)
        crop_bb = np.array((t, l, b, r))

        if allow_outside_center:
            # All bboxes are allowed.
            mask = np.ones(bbox.shape[0], dtype=bool)
        else:
            # Only bboxes that contain center are allowd.
            center = (bbox[:, :2] + bbox[:, 2:]) / 2
            mask = np.logical_and(crop_bb[:2] <= center, center < crop_bb[2:])\
                .all(axis=1)

        # Crop
        original_bbox, bbox = bbox, bbox.copy()
        bbox[:, :2] = np.maximum(bbox[:, :2], crop_bb[:2])
        bbox[:, 2:] = np.minimum(bbox[:, 2:], crop_bb[2:])

        truncated_mask = np.any(original_bbox != bbox, axis=1)

        # Adjust each coordinate to 0-based.
        bbox[:, :2] -= crop_bb[:2]
        bbox[:, 2:] -= crop_bb[:2]

        mask = np.logical_and(mask, (bbox[:, :2] < bbox[:, 2:]).all(axis=1))
        bbox = bbox[mask]
        truncated_mask = truncated_mask[mask]

        index = np.flatnonzero(mask)
        truncated_index = np.flatnonzero(truncated_mask)
        return bbox, {
            'index': index,
            'truncated_index': truncated_index,
        }

    @staticmethod
    def _slice_to_bounds(slice_):
        """Convert slice to tuple.
        Args:
            slice_ (slice): slice object.
        """
        if slice_ is None:
            return 0, np.inf

        if slice_.start is None:
            start = 0
        else:
            start = slice_.start

        if slice_.stop is None:
            end = np.inf
        else:
            end = slice_.stop

        return start, end


class ResizeWithBbox(object):
    """Transform class of Resize.
    Args:
        size (int or tuple):
            Resized size. If int is passed, size means both height and width.
        interpolation: Determines sampling strategy. This is one of
            ``PIL.Image.NEAREST``, ``PIL.Image.BILINEAR`*,
            ``PIL.Image.BICUBIC``, ``PIL.Image.LANCZOS``.
            Bilinear interpolation is the default strategy.
    """
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.interpolation = interpolation
        if isinstance(size, int):
            self.size = (size, size)
        elif isinstance(size, tuple):
            self.size = size
        else:
            raise TypeError("size must be int or tuple")

    def __call__(self, data):
        img, boxes, *rest_label = data
        h, w, _ = img.shape
        img = Image.fromarray(img.astype(np.uint8))
        img = img.resize(self.size, resample=self.interpolation)
        img = np.array(img).astype(np.float32)
        boxes = resize_bbox(boxes, (h, w), self.size)
        return (img, boxes, *rest_label)


class ToTensorWithBbox(object):
    """Transform class of Tensor.
    Args:
        device (str): Cuda device id.
    """
    def __call__(self, data):
        img, boxes, *rest_label = data
        img = torch.from_numpy(img).permute(2, 0, 1)
        boxes = torch.from_numpy(boxes)
        rest_label = [torch.from_numpy(label) for label in rest_label]
        return (img, boxes, *rest_label)


class NormalizeWithBbox(object):
    """Transform class of Normalize.
    Args:
        mean (list): Mean of each RGB pixel value.
        std (list): Standard deviation of each RGB pixel value.
    """
    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = std

    def __call__(self, data):
        img, boxes, *rest_label = data
        img = img.astype(np.float32)
        img -= self.mean
        img /= self.std
        return (img.astype(np.float32), boxes, *rest_label)
