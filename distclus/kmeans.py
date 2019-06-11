from distclus import bind
from .ffi import lib
from .oc import OnlineClust


class KMeans(OnlineClust):
    """Proxy a KMEANS algorithm implemented in native library"""

    def __init__(
            self, space='vectors', par=True, init='kmeanspp',
            k=16, nb_iter=100, frame_size=None, seed=None, data=None,
            inner_space=None, window=None
    ):
        super(KMeans, self).__init__(
            lib.KMeans, space, data, bind.par(par), bind.initializer(init), bind.none2zero(seed),
            k, nb_iter, bind.none2zero(frame_size), bind.none2zero(inner_space), bind.none2zero(window)
        )

    @property
    def iterations(self):
        figure = lib.RuntimeFigure(self.descr, lib.F_ITERATIONS)
        if figure.err:
            raise RuntimeError(figure.err)
        return figure.value
