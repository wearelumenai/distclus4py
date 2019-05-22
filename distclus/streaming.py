import numpy as np

from distclus import bind
from .ffi import lib
from .oc import OnlineClust


class Streaming(OnlineClust):
    """Proxy a Streaming algorithm implemented in native library"""

    def __init__(
            self, space='vectors', par=True,
            buffer_size=0, b=.95, lambd=3., seed=None, data=np.empty([0, 0]),
            inner_space=0, window=10
    ):
        super(Streaming, self).__init__(
            lib.Streaming, space, data, bind.seed(seed), buffer_size, b, lambd,
            inner_space, window
        )

