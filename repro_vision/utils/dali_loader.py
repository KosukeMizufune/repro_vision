import math

from nvidia.dali import ops, types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIClassificationIterator
import numpy as np


class DaliIterator:
    def __init__(self, data, batchsize=32):
        self.data = data
        self.batchsize = batchsize

    def __iter__(self):
        self.i = 0
        self.n = len(self.data)
        return self

    def __next__(self):
        batch = []
        labels = []
        for _ in range(self.batchsize):
            if self.data.train and self.i % self.n == 0:
                shuffled_idx = \
                    np.random.choice(np.arange(self.n), self.n, replace=False)
                self.data.data = self.data.data[shuffled_idx]
                self.data.targets = self.data.targets[shuffled_idx]
            img, label = self.data[self.i]
            batch.append(img)
            labels.append(label)
            self.i = (self.i + 1) % self.n
        return (batch, labels)


class SimplePipeline(Pipeline):
    def __init__(self, data, batchsize, num_threads, device_id):
        super(SimplePipeline, self).__init__(batchsize, num_threads, device_id, seed=12 + device_id)
        self.batchsize = batchsize
        self.iterator = iter(DaliIterator(data, batchsize))
        self.input = ops.ExternalSource()
        self.input_label = ops.ExternalSource()
        self.data = data
        self.res = ops.Resize(device="gpu", resize_x=224, resize_y=224,
                              interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            image_type=types.RGB
                                            )
        self.coin = ops.CoinFlip(probability=0.5)

    def iter_setup(self):
        (images, labels) = next(self.iterator)
        self.feed_input(self.jpegs, images, layout='HWC')
        self.feed_input(self.labels, labels)

    def define_graph(self):
        self.jpegs = self.input()
        self.labels = self.input_label()
        output = self.jpegs

        output = self.res(output.gpu())
        output = self.cmnp(output)
        return [output, self.labels.gpu()]


class DaliDataLoader:
    def __init__(self, pipe, size):
        pipe.build()
        self.dataset = pipe.data
        self.iterator = DALIClassificationIterator(pipe, size)

    def __iter__(self):
        return self

    def __len__(self):
        return int(math.ceil(self.iterator._size / self.iterator.batch_size))

    def __next__(self):
        try:
            data = next(self.iterator)
        except StopIteration:
            print('Resetting DALI loader')
            self.iterator.reset()
            raise StopIteration

        # Decode the data output
        input = data[0]['data'] / 255.
        target = data[0]['label'].squeeze().long()

        return input, target
