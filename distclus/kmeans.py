from distclus import bind
from .ffi import lib
from .oc import OnlineClust


class KMeans(OnlineClust):
    """Proxy a KMEANS algorithm implemented in native library"""

    def __init__(
            self, space='vectors', par=True, init='kmeanspp',
            k=16, mcmc_iter=100, frame_size=0, seed=None, data=None,
            inner_space=0, window=10
    ):
        super(KMeans, self).__init__(
            lib.KMeans, space, data, bind.par(par), bind.initializer(init), bind.seed(seed),
            k, mcmc_iter, frame_size, inner_space, window
        )

    @property
    def iterations(self):
        figure = lib.RuntimeFigure(self.descr, lib.F_ITERATIONS)
        if figure.err:
            raise RuntimeError(figure.err)
        return figure.value
