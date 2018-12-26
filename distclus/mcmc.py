import random
import weakref

from . import bind
from .ffi import lib

from .oc import OnlineClust

import numpy as np


class MCMC(OnlineClust):
    """Proxy a MCMC algorithm implemented in native library"""

    def __init__(
        self, dim=0, space='vectors', par=True, init='kmeanspp',
        init_k=8, max_k=16, mcmc_iter=100, frame_size=0, b=1,
        amp=0.1, norm=2, nu=3, init_iter=1, seed=None, data=np.empty([0, 0]),
        inner_space=0, window=10
    ):
        super(MCMC, self).__init__(
            space, par, init, seed, data,
            dim, init_k, max_k, mcmc_iter, frame_size, b, amp, norm, nu,
            init_iter, inner_space, window
        )

    @property
    def iterations(self):
        return lib.RuntimeFigure(self.descr, lib.F_ITERATIONS)
