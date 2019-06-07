from distclus import bind
from .ffi import lib
from .oc import OnlineClust


class MCMC(OnlineClust):
    """Proxy a MCMC algorithm implemented in native library"""

    def __init__(
            self, space='vectors', par=True, init='kmeanspp',
            init_k=8, max_k=16, mcmc_iter=100, frame_size=None, b=1.,
            amp=1., dim=None, nu=3., norm=2., seed=None,
            data=None, inner_space=None, window=None
    ):
        super(MCMC, self).__init__(
            lib.MCMC, space, data, bind.par(par), bind.initializer(init), bind.none2zero(seed),
            bind.none2zero(dim), init_k, max_k, mcmc_iter, bind.none2zero(frame_size), b, amp, norm, nu,
            bind.none2zero(inner_space), bind.none2zero(window)
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
