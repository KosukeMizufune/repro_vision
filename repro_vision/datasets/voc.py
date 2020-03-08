import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from torchvision.datasets.vision import VisionDataset
from PIL import Image


class VOCDetection(VisionDataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Detection Dataset.
    This dataset class constains ``(image, bbox, label, difficult)``.
    Note, bbox coordinates are 0-based.
    Args:
        root (string): Root directory of the VOC Dataset.
        year (string, optional): The dataset year, supports years 2007 to 2012.
        image_set (string, optional):
            Select the image_set to use, ``train``, ``trainval`` or ``val``
        transform (callable, optional):
            A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, required):
            A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional):
            A function/transform that takes input sample and its target as
            entry and returns a transformed version.
    """
    def __init__(self,
                 root,
                 year='2012',
                 image_set='train',
                 use_difficult=False,
                 return_difficult=False,
                 transform=None,
                 target_transform=None,
                 transforms=None):
        super(VOCDetection, self).__init__(root, transforms, transform,
                                           target_transform)
        if year == '2007' and image_set == 'test':
            year = '2007_test'
        self.year = year
        self.image_set = image_set
        self.use_difficult = use_difficult
        self.return_difficult = return_difficult

        base_dir = f"VOCdevkit/VOC{year}"
        voc_root = Path(self.root) / base_dir
        image_dir = voc_root / 'JPEGImages'
        annotation_dir = voc_root / 'Annotations'

        split_file = voc_root / f'ImageSets/Main/{image_set}.txt'

        with open(split_file, "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [image_dir / f"{f_name}.jpg" for f_name in file_names]
        self.annotations = [annotation_dir / f"{f_name}.xml"
                            for f_name in file_names]
        assert (len(self.images) == len(self.annotations))
        self.label_names = (
            'aeroplane',
            'bicycle',
            'bird',
            'boat',
            'bottle',
            'bus',
            'car',
            'cat',
            'chair',
            'cow',
            'diningtable',
            'dog',
            'horse',
            'motorbike',
            'person',
            'pottedplant',
            'sheep',
            'sofa',
            'train',
            'tvmonitor')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns: tuple:
            (image, bbox, label, difficult). Coordinates of bbox are 0-based.
        """
        img = Image.open(self.images[index]).convert('RGB')
        img = np.array(img)
        bbox, label, difficult = self._get_annotations(index)

        if self.transforms is not None:
            img, bbox, label, difficult = \
                self.transforms(img, bbox, label, difficult)
        if self.return_difficult:
            return img, bbox, label, difficult
        return img, bbox, label

    def __len__(self):
        return len(self.images)

    def _get_annotations(self, i):
        anno = ET.parse(self.annotations[i])
        bboxes = []
        labels = []
        difficults = []
        for obj in anno.findall('object'):
            # when in not using difficult split, and the object is
            # difficult, skipt it.
            is_difficult = int(obj.find('difficult').text)
            if not self.use_difficult and is_difficult == 1:
                continue
            difficults.append(is_difficult)

            bndbox_anno = obj.find('bndbox')
            # subtract 1 to make pixel indexes 0-based
            bboxes.append([
                int(bndbox_anno.find(tag).text) - 1
                for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
            name = obj.find('name').text.lower().strip()
            labels.append(self.label_names.index(name))
        bboxes = np.stack(bboxes).astype(np.float32)
        labels = np.stack(labels).astype(np.int)
        difficults = np.stack(difficults).astype(np.int)
        return bboxes, labels, difficults
