import random
import weakref

from . import bind

from .oc import OnlineClust

import numpy as np


class KMEANS(OnlineClust):
    """Proxy a KMEANS algorithm implemented in native library"""

    def __init__(
        self, space='vectors', par=True, init='kmeanspp',
        k=16, iter=100, frame_size=10000, seed=None, data=np.empty([0, 0]),
        inner_space=0, window=10
    ):
        super(KMEANS, self).__init__(
            space, par, init, seed, data, k, iter, frame_size,
            inner_space, window
        )
