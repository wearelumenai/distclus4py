import random
import weakref

from . import bind
from .ffi import lib


class MCMC:
    """
    Proxy a MCMC algorithm implemented in native library
    """

    def __init__(self, dim,
                 init_k=8, mcmc_iter=100, frame_size=10000,
                 b=1, amp=1, norm=2, nu=3,
                 init_iter=1, init="kmeanspp", seed=None):
        seed = seed or random.randint(0, 2 ** 63)
        descr = lib.MCMC(dim, init_k, mcmc_iter, frame_size, b, amp, norm, nu, init_iter, bind.initializer(init), seed)
        self.descr = descr
        self.__finalize = weakref.finalize(self, lambda: lib.FreeMCMC(descr))

    def fit(self, data):
        self.push(data)
        self.run()
        self.close()

    def push(self, data):
        arr, l1, l2 = bind.to_c_2d_array(data)
        lib.MCMCPush(self.descr, arr, l1, l2)

    def run(self, rasync=False):
        lib.MCMCRun(self.descr, 1 if rasync else 0)

    def close(self):
        lib.MCMCClose(self.descr)

    def predict(self, data, push=False):
        arr, l1, l2 = bind.to_c_2d_array(data)
        result = lib.MCMCPredict(self.descr, arr, l1, l2, 1 if push else 0)
        return bind.to_managed_1d_array(result)

    def centroids(self):
        result = lib.MCMCRealCentroids(self.descr)
        return bind.to_managed_2d_array(result)
