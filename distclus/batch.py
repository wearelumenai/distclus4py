import contextlib
import itertools as it
import numpy as np
from collections import deque


class Batch:
    def __init__(self, builder, frame_size=None, **kwargs):
        self._builder = builder
        self._kwargs = kwargs
        self._algo = self._builder(**self._kwargs)
        self._do_not_gc = None
        if frame_size:
            self.frame = deque([], frame_size)
        else:
            self.frame = None

    def run(self, rasync=True):
        return contextlib.closing(self)

    def push(self, data):
        # hack : scikit-learn compliance,
        # fit reruns algorithm initialization
        # thus previous result must not be garbage collected
        self._do_not_gc = self._algo
        self._algo = self._builder(**self._kwargs)
        self._kwargs['init'] = self._algo
        self._fit_frame(data)

    def _fit_frame(self, data):
        if self.frame is not None:
            self.frame.extend(data)
            self.fit(np.array(self.frame))
        else:
            self.fit(data)

    def fit(self, data):
        self._algo.fit(data)

    def predict(self, data):
        return self._algo.predict(data)

    def predict_online(self, data):
        return self._algo.predict_online(data)

    def close(self):
        pass

    @property
    def centroids(self):
        return self._algo.centroids

    @property
    def cluster_centers_(self):
        return self.centroids


def chunker(iterable, n):
    """Get chunks of fixed size from an iterable
    :param iterable: the iterable
    :param n: chunk size
    :return: a generator of chunks
    """
    gen = iter(iterable)
    while True:
        current = list(it.islice(gen, n))
        if current:
            yield current
        else:
            return
