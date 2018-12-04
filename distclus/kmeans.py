import random
import weakref

from . import bind

from .oc import OnlineClust


class KMEANS(OnlineClust):
    """Proxy a KMEANS algorithm implemented in native library"""

    def __init__(
        self, space='real', par=True, init='kmeanspp',
        k=16, iter=100, frame_size=10000, seed=None
    ):
        super(KMEANS, self).__init__(
            space, par, init, seed, k, iter, frame_size, seed
        )
