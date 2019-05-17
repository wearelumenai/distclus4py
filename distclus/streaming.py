import numpy as np

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
            space, par, 'given', seed, data, buffer_size, b, lambd,
            inner_space, window
        )

