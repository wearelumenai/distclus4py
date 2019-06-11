from distclus import bind
from .ffi import lib
from .oc import OnlineClust


class Streaming(OnlineClust):
    """Proxy a Streaming algorithm implemented in native library"""

    def __init__(
            self, space='vectors',
            buffer_size=1000, b=.95, lambd=3., seed=None, data=None,
            inner_space=0, window=10
    ):
        super(Streaming, self).__init__(
            lib.Streaming, space, data, bind.none2zero(seed), buffer_size, b, lambd,
            inner_space, window
        )

