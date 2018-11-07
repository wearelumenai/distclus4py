import random
import weakref

from . import bind
from .ffi import lib


class MCMC:
    """
    Proxy a MCMC algorithm implemented in native library
    """

    def __init__(self, dim,
                 init_k=8, mcmc_iter=30, framesize=10000,
                 b=1, amp=1, norm=2, nu=1,
                 init_iter=1, init="kmeanspp", seed=None):
        seed = seed or random.randint(0, 2 ** 63)
        self.algo = lib.MCMC(dim, init_k, mcmc_iter, framesize, b, amp, norm, nu, init_iter, bind.initializer(init),
                             seed)
        self.__finalize = weakref.finalize(self, self.__cleanup)

    def fit(self, data):
        self.push(data)
        self.run()
        self.close()

    def push(self, data):
        arr, l1, l2 = bind.to_c_2d_array(data)
        lib.MCMCPush(self.algo, arr, l1, l2)

    def run(self, rasync=False):
        lib.MCMCRun(self.algo, 1 if rasync else 0)

    def close(self):
        lib.MCMCClose(self.algo)

    def predict(self, data, push=False):
        arr, l1, l2 = bind.to_c_2d_array(data)
        result = lib.MCMCPredict(self.algo, arr, l1, l2, 1 if push else 0)
        return bind.to_managed_1d_array(result)

    def centroids(self):
        result = lib.MCMCRealCentroids(self.algo)
        return bind.to_managed_2d_array(result)

    def __cleanup(self):
        lib.FreeMCMC(self.algo)
