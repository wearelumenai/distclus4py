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
            inner_space='vectors', window=10
    ):
        super(Streaming, self).__init__(
            lib.Streaming, space, data, bind.none2zero(seed),
            buffer_size, mu,
            sigma, outRatio, outAfter,
            iter, iter_freq, data_per_iter, timeout,
            bind.space(inner_space), window
        )

    def fit(self, data, iter=0, duration=0):
        """
        Sequentially push train data, run in synchronous mode
        and close the algorithm.
        :param data: train data
        :return: centroids
        """
        self._set_descr()
        self.push(data)
        if iter == 0:
            iter = len(data)
        return self.batch(iter, duration)

    @property
    def max_distance(self):
        """
        Get the number of iterations done so far
        """
        return self._figure(lib.F_MAX_DISTANCE)
