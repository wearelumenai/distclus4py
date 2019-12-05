from distclus import bind
from .ffi import lib
from .oc import OnlineClust


class Streaming(OnlineClust):
    """Proxy a Streaming algorithm implemented in native library"""

    def __init__(
            self, space='vectors',
            buffer_size=100, mu=.5, sigma=0.1, outRatio=2., outAfter=5,
            seed=None,
            iter=0, iter_freq=0, data_per_iter=0, timeout=0,
            data=None,
            inner_space=0, window=10
    ):
        super(Streaming, self).__init__(
            lib.Streaming, space, data, bind.none2zero(seed), buffer_size, mu,
            sigma, outRatio, outAfter,
            iter, iter_freq, data_per_iter, timeout,
            inner_space, window
        )

    @property
    def max_distance(self):
        """
        Get the number of iterations done so far
        """
        figure = lib.RuntimeFigure(self.descr, lib.F_MAX_DISTANCE)
        if figure.err:
            raise RuntimeError(figure.err)
        return figure.value
