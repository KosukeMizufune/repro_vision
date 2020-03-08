# flake8: noqa: W605
import json
import itertools
import sys

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import nms
import numpy as np

from repro_vision.serializer import load_pth
from repro_vision.utils.bbox_utils import resize_bbox, bbox_iou

_imagenet_mean = torch.Tensor([104, 117, 123]).reshape((-1, 1, 1))


class MultiboxCoder(object):
    """A helper class to encode/decode bounding boxes.
    This class encodes :obj:`(bbox, label)` to :obj:`(mb_loc, mb_label)`
    and decodes :obj:`(mb_loc, mb_conf)` to :obj:`(bbox, label, score)`.
    These encoding/decoding are used in Single Shot Multibox Detector [#]_.
    * :obj:`mb_loc`: An array representing offsets and scales \
        from the default bounding boxes. \
        Its shape is :math:`(K, 4)`, where :math:`K` is the number of \
        the default bounding boxes. \
        The second axis is composed by \
        :math:`(\Delta y, \Delta x, \Delta h, \Delta w)`. \
        These values are computed by the following formulas.
        * :math:`\Delta y = (b_y - m_y) / (m_h * v_0)`
        * :math:`\Delta x = (b_x - m_x) / (m_w * v_0)`
        * :math:`\Delta h = log(b_h / m_h) / v_1`
        * :math:`\Delta w = log(b_w / m_w) / v_1`
        :math:`(m_y, m_x)` and :math:`(m_h, m_w)` are \
        center coodinates and size of a default bounding box. \
        :math:`(b_y, b_x)` and :math:`(b_h, b_w)` are \
        center coodinates and size of \
        a given bounding boxes that is assined to the default bounding box. \
        :math:`(v_0, v_1)` are coefficients that can be set \
        by argument :obj:`variance`.
    * :obj:`mb_label`: An array representing classes of \
        ground truth bounding boxes. Its shape is :math:`(K,)`.
    * :obj:`mb_conf`: An array representing classes of \
        predicted bounding boxes. Its shape is ``(K, n_class + 1)``.
    .. [#] Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy,
       Scott Reed, Cheng-Yang Fu, Alexander C. Berg.
       SSD: Single Shot MultiBox Detector. ECCV 2016.
    Args:
        grids (iterable of ints): An iterable of integers.
            Each integer indicates the size of a feature map.
        aspect_ratios (iterable of tuples of ints):
            An iterable of tuples of integers
            used to compute the default bounding boxes.
            Each tuple indicates the aspect ratios of
            the default bounding boxes at each feature maps.
            The length of this iterable should be :obj:`len(grids)`.
        steps (iterable of floats): The step size for each feature map.
            The length of this iterable should be :obj:`len(grids)`.
        sizes (iterable of floats): The base size of default bounding boxes
            for each feature map.
            The length of this iterable should be :obj:`len(grids) + 1`.
        variance (tuple of floats): Two coefficients for encoding/decoding
            the locations of bounding boxes. The first value is used to
            encode/decode coordinates of the centers.
            The second value is used to encode/decode the sizes of
            bounding boxes.
    """

    def __init__(self, grids, aspect_ratios, steps, sizes, variance):
        if not len(aspect_ratios) == len(grids):
            raise ValueError('The length of aspect_ratios is wrong.')
        if not len(steps) == len(grids):
            raise ValueError('The length of steps is wrong.')
        if not len(sizes) == len(grids) + 1:
            raise ValueError('The length of sizes is wrong.')

        default_bbox = []

        for k, grid in enumerate(grids):
            for v, u in itertools.product(range(grid), repeat=2):
                cy = (v + 0.5) * steps[k]
                cx = (u + 0.5) * steps[k]

                s = sizes[k]
                default_bbox.append((cy, cx, s, s))

                s = np.sqrt(sizes[k] * sizes[k + 1])
                default_bbox.append((cy, cx, s, s))

                s = sizes[k]
                for ar in aspect_ratios[k]:
                    default_bbox.append(
                        (cy, cx, s / np.sqrt(ar), s * np.sqrt(ar)))
                    default_bbox.append(
                        (cy, cx, s * np.sqrt(ar), s / np.sqrt(ar)))

        # (center_y, center_x, height, width)
        self._default_bbox = torch.Tensor(default_bbox)
        self._variance = variance

    def to(self, device):
        self._default_bbox = self._default_bbox.to(device)

    def encode(self, bbox, label, iou_thresh=0.5):
        """Encodes coordinates and classes of bounding boxes.
        This method encodes :obj:`bbox` and :obj:`label` to :obj:`mb_loc`
        and :obj:`mb_label`, which are used to compute multibox loss.
        Args:
            bbox (torch.Tensor): A float tensor of shape :math:`(R, 4)`,
                where :math:`R` is the number of bounding boxes in an image.
                Each bounding box is organized by
                :math:`(y_{min}, x_{min}, y_{max}, x_{max})`
                in the second axis.
            label (torch.Tensor) : An integer array of shape :math:`(R,)`.
                Each value indicates the class of the bounding box.
            iou_thresh (float): The threshold value to determine
                a default bounding box is assigned to a ground truth
                or not. The default value is :obj:`0.5`.
        Returns:
            tuple of two tensors:
            This method returns a tuple of two tensors,
            :obj:`(mb_loc, mb_label)`.
            * **mb_loc**: A float tensor of shape :math:`(K, 4)`, \
                where :math:`K` is the number of default bounding boxes.
            * **mb_label**: An integer tensor of shape :math:`(K,)`.
        """
        device = bbox.device
        if len(bbox) == 0:
            return (
                torch.zeros(self._default_bbox.shape).to(bbox),
                torch.zeros(self._default_bbox.shape[0]).to(label))

        # Because function bbox_iou takes (y_min, x_min, y_max, x_max),
        # _default_bbox must be transformed.
        iou = bbox_iou(
            torch.cat((self._default_bbox[:, :2] - self._default_bbox[:, 2:] / 2,  # noqa
                       self._default_bbox[:, :2] + self._default_bbox[:, 2:] / 2), # noqa
                      dim=1),
            bbox)

        # initialize with -1 (background) for now.
        # 1st argument of torch.full must be tuple.
        index = torch.full((len(self._default_bbox),), -1,
                           dtype=int, device=device)

        # Save index of column and row with high iou order.
        # If IoU becomes smaller than threshold, save loop gets interupted.
        masked_iou = iou.clone()
        ncol = masked_iou.shape[1]
        while True:
            # Below three lines mean np.unravel_index.
            max_index = masked_iou.argmax()
            row = max_index / ncol
            col = max_index % ncol
            if masked_iou[row, col] <= 1e-6:
                break
            index[row] = col
            masked_iou[row, :] = 0
            masked_iou[:, col] = 0

        # Collect indexes which are not saved above process
        # i.e. Index that has second or smaller highest IoU.
        # iou.max -> namedtupel (value, indices)

        mask = (index < 0) * (iou.max(dim=1)[0] >= iou_thresh)  # logical_and
        if mask.any():
            index[mask] = iou[mask].argmax(dim=1)

        mb_bbox = bbox[index].clone()
        # (y_min, x_min, y_max, x_max) -> (y_min, x_min, height, width)
        mb_bbox[:, 2:] -= mb_bbox[:, :2]
        # (y_min, x_min, height, width) -> (center_y, center_x, height, width)
        mb_bbox[:, :2] += mb_bbox[:, 2:] / 2

        mb_loc = torch.empty_like(mb_bbox)
        mb_loc[:, :2] = (mb_bbox[:, :2] - self._default_bbox[:, :2]) \
            / (self._variance[0] * self._default_bbox[:, 2:])
        mb_loc[:, 2:] = torch.log(mb_bbox[:, 2:] / self._default_bbox[:, 2:]) \
            / self._variance[1]

        # [0, n_fg_class - 1] -> [1, n_fg_class]
        mb_label = label[index] + 1
        # 0 is for background
        mb_label[index < 0] = 0

        return mb_loc.to(bbox), mb_label.to(label)

    def decode(self, mb_loc, mb_conf, nms_thresh=0.45, score_thresh=0.6):
        """Decodes back to coordinates and classes of bounding boxes.
    
        This method decodes :obj:`mb_loc` and :obj:`mb_conf` returned
        by a SSD network back to :obj:`bbox`, :obj:`label` and :obj:`score`.
        Args:
            mb_loc (torch.Tensor): A float array whose shape is
                :math:`(K, 4)`, :math:`K` is the number of
                default bounding boxes.
            mb_conf (torch.Tensor): A float array whose shape is
                ``(K, n_class + 1)``.
            nms_thresh (float): The threshold value
                for :func:`~chainercv.utils.non_maximum_suppression`.
                The default value is :obj:`0.45`.
            score_thresh (float): The threshold value for confidence score.
                If a bounding box whose confidence score is lower than
                this value, the bounding box will be suppressed.
                The default value is :obj:`0.6`.
        Returns:
            tuple of three tensors:
            This method returns a tuple of three tensors,
            :obj:`(bbox, label, score)`.
            * **bbox**: A float tensor of shape :math:`(R, 4)`, \
                where :math:`R` is the number of bounding boxes in a image. \
                Each bounding box is organized by \
                :math:`(y_{min}, x_{min}, y_{max}, x_{max})` \
                in the second axis.
            * **label** : An integer tensor of shape :math:`(R,)`. \
                Each value indicates the class of the bounding box.
            * **score** : A float tensor of shape :math:`(R,)`. \
                Each value indicates how confident the prediction is.
        """
        device = mb_loc.device
        # (center_y, center_x, height, width)
        mb_bbox = self._default_bbox.clone()
        mb_bbox[:, :2] += mb_loc[:, :2] * self._variance[0] \
            * self._default_bbox[:, 2:]
        mb_bbox[:, 2:] *= torch.exp(mb_loc[:, 2:] * self._variance[1])

        # (center_y, center_x, height, width) -> (y_min, x_min, height, width)
        mb_bbox[:, :2] -= mb_bbox[:, 2:] / 2
        # (center_y, center_x, height, width) -> (y_min, x_min, y_max, x_max)
        mb_bbox[:, 2:] += mb_bbox[:, :2]

        # softmax
        mb_score = torch.exp(mb_conf)
        mb_score /= mb_score.sum(dim=1, keepdim=True)

        bbox = []
        label = []
        score = []
        for l in range(mb_conf.shape[1] - 1):
            bbox_l = mb_bbox
            # the l-th class corresponds for the (l + 1)-th column.
            score_l = mb_score[:, l + 1]

            mask = score_l >= score_thresh
            bbox_l = bbox_l[mask]
            score_l = score_l[mask]

            if nms_thresh is not None:
                indices = nms(bbox_l, score_l, nms_thresh)
                bbox_l = bbox_l[indices]
                score_l = score_l[indices]

            bbox.append(bbox_l)
            label.append(torch.Tensor((l,) * len(bbox_l)).to(device))
            score.append(score_l)

        bbox = torch.cat(bbox, dim=0).to(torch.float32)
        label = torch.cat(label, dim=0).to(torch.long)
        score = torch.cat(score, dim=0).to(torch.float32)
        return bbox, label, score


class SSD(nn.Module):
    """
    Base class of Single Shot Multibox Detector.
    This is a base class of Single Shot Multibox Detector [#]_.
    .. [#] Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy,
        Scott Reed, Cheng-Yang Fu, Alexander C. Berg.
        SSD: Single Shot MultiBox Detector. ECCV 2016.
    Args:
        extractor (torch.nn.Module): A link which extracts feature maps.
        multibox (torch.nn.Module): A link which computes :obj:`mb_locs` \
        config_multibox (dict): This argument is used by \
            :class:`~src.models.multibox.MultiboCoder`.
        """
    def __init__(self, extractor, multibox, coder, mean=0):
        super(SSD, self).__init__()
        self.extractor = extractor
        self.multibox = multibox
        self.coder = coder
        self.mean = mean
        self.nms_thresh = 0.45
        self.score_thresh = 0.01

    def forward(self, x):
        """Compute localization and classification from a batch of images.
        This method computes two variables, :obj:`mb_locs` and :obj:`mb_confs`.
        These variables are also used in training SSD.
        Args:
            x (torch.Tensor): A variable holding a batch of images.
                The images are preprocessed by :meth:`_prepare`.
        Returns:
            tuple of torch.Tensor:
            This method returns two variables, :obj:`mb_locs` and
            :obj:`mb_confs`.
            * **mb_locs**: A variable of float arrays of shape \
                ``(B, K, 4)``, \
                where :math:`B` is the number of samples in the batch and \
                :math:`K` is the number of default bounding boxes.
            * **mb_confs**: A variable of float arrays of shape \
                ``(B, K, n_class + 1)``.
        """
        self._check_input_size(x)
        return self.multibox(self.extractor(x))

    def _prepare(self, img):
        img = F.interpolate(img.unsqueeze(0), self.in_size)[0]
        img -= self.mean.to(img)
        return img

    def to(self, *args, **kwargs):
        super(SSD, self).to(*args, **kwargs)
        device = next(self.parameters()).device
        self.coder.to(device)

    def use_preset(self, preset):
        """Use the given preset during prediction.
        This method changes values of :obj:`nms_thresh` and
        :obj:`score_thresh`. These values are a threshold value
        used for non maximum suppression and a threshold value
        to discard low confidence proposals in :meth:`predict`,
        respectively.
        If the attributes need to be changed to something
        other than the values provided in the presets, please modify
        them by directly accessing the public attributes.
        Args:
            preset ({'visualize', 'evaluate'}): A string to determine the
                preset to use.
        """

        if preset == 'visualize':
            self.nms_thresh = 0.45
            self.score_thresh = 0.6
        elif preset == 'evaluate':
            self.nms_thresh = 0.45
            self.score_thresh = 0.01
        else:
            raise ValueError('preset must be visualize or evaluate')

    def predict(self, imgs):
        """Predict method of SSD.
        First, :func:`self._prepare` resize and normalize input images.
        Second, :func:`self.forward` outputs ``locs`` and ``confs``. Next,
        :func:`self.coder.decode` converts these variables to bounding box
        coordinates and confidence scores. Finally, ``resize_bbox`` transforms
        those coordinates to coordinates of original resolution.
        Args:
            imgs (list of torch.Tensor): Input images.
        Returns:
            tuple of numpy.ndarray:
            This method returns three variables, :obj:`bboxes`,
            :obj:`labels`, and :obj:`scores`.
            * **bboxes**: A list of float arrays of shape \
                ``(K, 4)``, \
                where :math:`K` is the number of bounding boxes.
            * **labels**: A list of float arrays of shape ``(K)``.
            * **scores**: A list of float arrays of shape ``(K)``.
        """
        x = []
        sizes = []
        for img in imgs:
            _, height, width = img.shape
            img = self._prepare(img)
            x.append(img)
            sizes.append((height, width))
        x = torch.stack(x)

        locs, confs = self.forward(x)

        bboxes = []
        labels = []
        scores = []

        for mb_loc, mb_conf, size in zip(locs, confs, sizes):
            bbox, label, score = self.coder.decode(
                mb_loc, mb_conf, self.nms_thresh, self.score_thresh)
            bbox = bbox.cpu().numpy()
            label = label.cpu().numpy()
            score = score.cpu().numpy()
            bbox = resize_bbox(
                bbox, (self.in_size, self.in_size), size)
            bboxes.append(bbox)
            labels.append(label)
            scores.append(score)

        return bboxes, labels, scores

    def init_weight(self, correspondence_file, weight_file):
        with open(correspondence_file, "r", encoding="utf-8") as f:
            correspondence = json.load(f)
        load_pth(self, torch.load(weight_file), correspondence)     

    @staticmethod
    def _check_input_size(self, x):
        return NotImplementedError


class ReluConv(nn.Module):
    """Sequential operation of Convolution and ReLU.
    Args:
        in_ch (int): The size of input channel.
        out_ch (int): The size of output channel.
        k_size (int): Kernel size of convolution.
        stride (int): Stride size of convolution.
        pad (int): Padding size of convolution.
        dilation (int): Paramerter of Dilated Convolution. Default values is 1.
    """
    def __init__(self, in_ch, out_ch, k_size, stride, pad, dilation=1):
        super(ReluConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k_size, stride, pad, dilation)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.conv(x))


class Normalize(nn.Module):
    """Learnable L2 normalization [#]_.
    This link normalizes input along the channel axis and scales it.
    The scale factors are trained channel-wise.
    .. [#] Wei Liu, Andrew Rabinovich, Alexander C. Berg.
       ParseNet: Looking Wider to See Better. ICLR 2016.
    Args:
        n_channels (int): The number of channels.
    """
    def __init__(self, n_channels):
        super(Normalize, self).__init__()
        self.n_channels = n_channels
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(1).unsqueeze(2) * x
        return out


class VGG16(nn.Module):
    """An extended VGG-16 model for SSD
    This is an extended VGG-16 model proposed in [#]_.
    The differences from original VGG-16 [#]_ are shown below.
    * :obj:`conv5_1`, :obj:`conv5_2` and :obj:`conv5_3` are changed from \
    Normal Convolution to Dilated Convolution.
    * Normalize Module is inserted after :obj:`conv4_3`.
    * The parameters of max pooling after :obj:`conv5_3` are changed.
    * :obj:`fc6` and :obj:`fc7` are converted to :obj:`conv6` and :obj:`conv7`.
    .. [#] Wei Liu, Dragomir Anguelov, Dumitru Erhan,
       Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg.
       SSD: Single Shot MultiBox Detector. ECCV 2016.
    .. [#] Karen Simonyan, Andrew Zisserman.
       Very Deep Convolutional Networks for Large-Scale Image Recognition.
       ICLR 2015.
    """
    def __init__(self):
        super(VGG16, self).__init__()
        self.layer1 = nn.Sequential(
            ReluConv(3, 64, 3, 1, 1),
            ReluConv(64, 64, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ReluConv(64, 128, 3, 1, 1),
            ReluConv(128, 128, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ReluConv(128, 256, 3, 1, 1),
            ReluConv(256, 256, 3, 1, 1),
            ReluConv(256, 256, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            ReluConv(256, 512, 3, 1, 1),
            ReluConv(512, 512, 3, 1, 1),
            ReluConv(512, 512, 3, 1, 1),
        )
        self.l2norm = Normalize(512)
        self.layer2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ReluConv(512, 512, 3, 1, 1),
            ReluConv(512, 512, 3, 1, 1),
            ReluConv(512, 512, 3, 1, 1),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ReluConv(512, 1024, 3, 1, 6, 6),
            ReluConv(1024, 1024, 1, 1, 0),
        )

    def forward(self, x):
        x = self.layer1(x)
        h1 = self.l2norm(x)
        h2 = self.layer2(x)
        return [h1, h2]


class VGGExtractor300(VGG16):
    """A VGG-16 based feature extractor for SSD300.
    This is a feature extractor for SSD.
    """
    def __init__(self):
        super(VGGExtractor300, self).__init__()
        self.conv8_1 = nn.Conv2d(1024, 256, 1, 1, 0)
        self.conv8_2 = nn.Conv2d(256, 512, 3, 2, 1)

        self.conv9_1 = nn.Conv2d(512, 128, 1, 1, 0)
        self.conv9_2 = nn.Conv2d(128, 256, 3, 2, 1)

        self.conv10_1 = nn.Conv2d(256, 128, 1, 1, 0)
        self.conv10_2 = nn.Conv2d(128, 256, 3, 1, 0)

        self.conv11_1 = nn.Conv2d(256, 128, 1, 1, 0)
        self.conv11_2 = nn.Conv2d(128, 256, 3, 1, 0)

        self.activation = nn.ReLU()

    def forward(self, x):
        ys = super(VGGExtractor300, self).forward(x)
        for i in range(8, 11 + 1):
            h = ys[-1]
            h = self.activation(getattr(self, f'conv{i}_1')(h))
            h = self.activation(getattr(self, f'conv{i}_2')(h))
            ys.append(h)
        return ys


class MultiBox(nn.Module):
    """Multibox head of Single Shot Multibox Detector.
    This is a head part of Single Shot Multibox Detector [#]_.
    This link computes :obj:`mb_locs` and :obj:`mb_confs` from feature maps.
    :obj:`mb_locs` contains information of the coordinates of bounding boxes
    and :obj:`mb_confs` contains confidence scores of each classes.
    .. [#] Wei Liu, Dragomir Anguelov, Dumitru Erhan,
       Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg.
       SSD: Single Shot MultiBox Detector. ECCV 2016.
    Args:
        n_class (int): The number of classes possibly including the background.
        aspect_ratios (iterable of tuple or int): The aspect ratios of
            default bounding boxes for each feature map.
    """
    def __init__(self, n_class, aspect_ratios):
        super(MultiBox, self).__init__()
        in_chs = [512, 1024, 512, 256, 256, 256]
        loc_layers = []
        conf_layers = []
        for ar, in_ch in zip(aspect_ratios, in_chs):
            n = (len(ar) + 1) * 2
            loc_layers.append(nn.Conv2d(in_ch, n * 4, 3, padding=1))
            conf_layers.append(nn.Conv2d(in_ch, n * n_class, 3, padding=1))
        self.loc_layers = nn.ModuleList(loc_layers)
        self.conf_layers = nn.ModuleList(conf_layers)
        self.n_class = n_class

    def forward(self, hs):
        locs = []
        confs = []
        for loc_layer, conf_layer, h in zip(self.loc_layers, self.conf_layers, hs):  # noqa
            loc = loc_layer(h)
            loc = loc.permute((0, 2, 3, 1))
            loc = loc.reshape((loc.shape[0], -1, 4))
            locs.append(loc)

            conf = conf_layer(h)
            conf = conf.permute(0, 2, 3, 1)
            conf = conf.reshape(conf.shape[0], -1, self.n_class)
            confs.append(conf)
        locs = torch.cat(locs, dim=1)
        confs = torch.cat(confs, dim=1)
        return locs, confs


class SSD300(SSD):
    def __init__(self, n_class, multibox_coder, correspondence_file,
                 weight_file):
        """Single Shot Multibox Detector with 300x300 inputs.
        This is a model of Single Shot Multibox Detector [#]_.
        This model uses :class:`VGGExtractor300` as its feature extractor.
        .. [#] Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy,
        Scott Reed, Cheng-Yang Fu, Alexander C. Berg.
        SSD: Single Shot MultiBox Detector. ECCV 2016.
        """
        super(SSD300, self).__init__(
            extractor=VGGExtractor300(),
            multibox=MultiBox(
                n_class=n_class + 1,
                aspect_ratios=((2,), (2, 3), (2, 3), (2, 3), (2,), (2,))),
            config_multibox=multibox_coder,
            mean=_imagenet_mean)
        self.in_size = 300
        self.init_weight(correspondence_file, weight_file)

    @staticmethod
    def _check_input_size(x):
        if x.shape[2:] != torch.Size([300, 300]):
            raise ValueError("Input width and height must be 300, "
                             f"but got {x.shape[2:]}.")


def _elementwise_softmax_cross_entropy(conf, label):
    """Element wise softmax cross entropy.
    Args:
        conf (torch.Tensor): Predicted score of shape ``(n, n_b, n_c)``
            where ``n`` is batchsize, ``n_b`` is the number of multibox, and
            ``n_c`` is the number of class including background.
        label (torch.Tensor): Ground truth label of shape ``(n, n_b)`` where
            ``n`` is batchsize and ``n_b`` is the number of multibox.
    Returns:
        torch.Tensor: Loss of shape ``(n, n_b)`` where \
            ``n`` is batchsize and ``n_b`` is the number of multibox.
    """
    shape = label.shape
    conf = conf.reshape(-1, conf.shape[-1])
    label = label.flatten()
    return F.cross_entropy(conf, label, reduction='none').reshape(shape)


def _hard_negative(x, positive, k):
    """Hard Negative Mining.
    Args:
        x (torch.Tensor): Tensor of shape ``(n, n_b) where
            ``n`` is batchsize and ``n_b`` is the number of multibox.
        positive (torch.BoolTensor): Tensor of shape ``(n, n_b)`` where
            ``n`` is batchsize and ``n_b`` is the number of multibox.
        k (int or float): Parameter for hard negative mining.
    Returns:
        torch.BoolTensor: Flag of hard negative of shape ``(n, n_b)`` where
            ``n`` is batchsize and ``n_b`` is the number of multibox.
    """
    # rank negative loss with lower order
    rank = (x * (positive.long() - 1)).argsort(axis=1).argsort(axis=1)
    hard_negative = rank < (positive.sum(axis=1) * k).unsqueeze(1)
    return hard_negative


def multibox_loss(mb_locs, mb_confs, gt_mb_locs, gt_mb_labels, k):
    """MultiBox Loss.
    Args:
        mb_locs (torch.Tensor): Predicted locs of shape ``(b, n_b, 4)`` where
            ``n`` is batchsize and ``n_b`` is the number of multibox.
        mb_confs (torch.Tensor): Predicted confs of shape ``(b, n_b, n_c)``
            where ``n`` is batchsize, ``n_b`` is the number of multibox, and
            ``n_c`` is the number of class including background.
        gt_mb_locs (torch.Tensor): Ground truth locs of shape ``(b, n_b, 4)``
            where ``n`` is batchsize, ``n_b`` is the number of multibox.
        gt_mb_labels (torch.Tensor): Ground truth labels of shape
            ``(b, n_b)`` where ``n`` is batchsize, ``n_b`` is the number
            of multibox.
        k (int or float): Parameter for hard negative mining.
    Returns:
        tuple: Localization loss and Confidence loss of torch.Tensor (scalar).
    """
    positive = gt_mb_labels > 0  # 0 is background.
    n_positive = positive.sum()
    loc_loss = F.smooth_l1_loss(mb_locs, gt_mb_locs, reduction='none')
    loc_loss = torch.sum(loc_loss, dim=-1)
    loc_loss *= positive
    loc_loss = torch.sum(loc_loss) / n_positive

    conf_loss = _elementwise_softmax_cross_entropy(mb_confs, gt_mb_labels)
    hard_negative = _hard_negative(conf_loss, positive, k)
    # logical or
    conf_loss *= (positive + hard_negative)
    conf_loss = torch.sum(conf_loss) / n_positive

    return loc_loss, conf_loss


class MultiboxLoss(nn.Module):
    """Loss Class of Multibox Loss.
    Args:
        alpha (float): The weight term between confidnece loss and
            localization loss.
        k (int or float): Parameter for hard negative mining.
    """
    def __init__(self, coder, alpha=0.1, k=3):
        super(MultiboxLoss, self).__init__()
        self.k = k
        self.alpha = alpha
        self.coder = coder

    def forward(self, preds, bboxes, labels):
        mb_locs, mb_confs = preds
        gt_mb_locs, gt_mb_labels = self.coder.encode(bboxes, labels)
        loc_loss, class_loss = multibox_loss(mb_locs, mb_confs, gt_mb_locs,
                                             gt_mb_labels, self.k)
        loss = loc_loss * self.alpha + class_loss
        return loss


def get_model(net_name, net_params, loss_params, coder_params, logger=None):
    coder = MultiboxCoder(**coder_params)
    net = getattr(sys.modules[__name__], net_name)(coder=coder, **net_params)
    loss = MultiboxLoss(coder=coder, **loss_params)
    return net, loss
