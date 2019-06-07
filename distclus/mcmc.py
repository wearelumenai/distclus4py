from distclus import bind
from .ffi import lib
from .oc import OnlineClust


class MCMC(OnlineClust):
    """Proxy a MCMC algorithm implemented in native library"""

    def __init__(
            self, dim=0, space='vectors', par=True, init='kmeanspp',
            init_k=8, max_k=16, mcmc_iter=100, frame_size=0, b=1,
            amp=0.1, norm=2, nu=3, init_iter=1, seed=None,
            data=None, inner_space=None, window=None
    ):
        super(MCMC, self).__init__(
            lib.MCMC, space, data, bind.par(par), bind.initializer(init), bind.none2zero(seed),
            dim, init_k, max_k, mcmc_iter, frame_size, b, amp, norm, nu,
            init_iter, bind.none2zero(inner_space), bind.none2zero(window)
        )

    @property
    def iterations(self):
        """
        Get the number of iterations done so far
        """
        figure = lib.RuntimeFigure(self.descr, lib.F_ITERATIONS)
        if figure.err:
            raise RuntimeError(figure.err)
        return figure.value

    @property
    def acceptations(self):
        """
        Get the number of iterations done so far
        """
        figure = lib.RuntimeFigure(self.descr, lib.F_ACCEPTATIONS)
        if figure.err:
            raise RuntimeError(figure.err)
        return figure.value
