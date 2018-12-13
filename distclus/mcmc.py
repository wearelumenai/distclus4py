import random
import weakref

from . import bind

from .oc import OnlineClust


class MCMC(OnlineClust):
    """Proxy a MCMC algorithm implemented in native library"""

    def __init__(
        self, dim=0, space='real', par=True, init='kmeanspp',
        init_k=8, max_k=16, mcmc_iter=100, frame_size=10000, b=1,
        amp=1, norm=2, nu=3, init_iter=1, seed=None
    ):
        super(MCMC, self).__init__(
            space, par, init, seed,
            dim, init_k, max_k, mcmc_iter, frame_size, b, amp, norm, nu,
            init_iter
        )
