import random
import weakref

from . import bind
from .ffi import lib

from .oc import OnlineClust

import numpy as np


class KMEANS(OnlineClust):
    """Proxy a KMEANS algorithm implemented in native library"""

    def __init__(
        self, space='vectors', par=True, init='kmeanspp',
        k=16, iter=100, frame_size=0, seed=None, data=np.empty([0, 0]),
        inner_space=0, window=10
    ):
        super(KMEANS, self).__init__(
            space, par, init, seed, data, k, iter, frame_size,
            inner_space, window
        )

    @property
    def iterations(self):
        return lib.RuntimeFigure(self.descr, lib.F_ITERATIONS)
