from distclus import bind
from .ffi import lib
from .oc import OnlineClust


class KMeans(OnlineClust):
    """Proxy a KMEANS algorithm implemented in native library"""

    def __init__(
            self, space='vectors', par=True, init='kmeanspp', init_descr=None,
            k=16, nb_iter=100, frame_size=None, seed=None,
            iter_freq=0, data_per_iter=0, timeout=0, num_cpu=0, iter=None,
            data=None, inner_space=None, window=None
    ):
        super(KMeans, self).__init__(
            lib.KMeans, space, data, bind.par(par),
            *bind.initializer(init), bind.none2zero(seed),
            k, nb_iter if iter is None else iter, bind.none2zero(frame_size),
            iter_freq, data_per_iter, timeout, num_cpu,
            bind.none2zero(inner_space),
            bind.none2zero(window)
        )
