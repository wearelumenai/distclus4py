import random
import weakref

from . import bind
from .ffi import lib

import numpy as np


class OnlineClust:
    """Base class for algorithm implementation using a native library"""

    def __init__(
        self,
        space='vectors', par=True, init='kmeanspp', seed=None,
        data=np.empty([0, 0]), *args
    ):
        space = bind.space(space)
        init = bind.initializer(init)
        seed = 0 if seed is None else seed
        par = 1 if par else 0
        arr, l1, l2 = bind.to_c_2d_array(data)
        self.args = [space, par, init, seed, arr, l1, l2] + list(args)
        self.reset()
        self.__finalize = weakref.finalize(self, self._free)

    def _free(self):
        if hasattr(self, 'descr'):
            lib.Free(self.descr)

    def _set_descr(self):
        descr = getattr(lib, self.__class__.__name__.upper())
        self.descr = descr(*self.args)

    def reset(self):
        """Reset the algorithm"""
        self._free()
        self._set_descr()

    def fit(self, data):
        """Execute sequentially push, run and close methods.

        :param data: data to process"""
        self.reset()
        self.push(data)
        err = self.run()
        if err == 0:
            return self.close()
        return err

    def push(self, data):
        """Push input data to process.

        :param data: data to push in the algorithme."""
        arr, l1, l2 = bind.to_c_2d_array(data)
        lib.Push(self.descr, arr, l1, l2)

    def __radd__(self, data):
        return self.push(data)

    def run(self, rasync=False):
        """Execute (a-)synchronously the alogrithm

        :param bool rasync: Asynchronous execution if True. Default is False.
        """
        return lib.Run(self.descr, 1 if rasync else 0)

    def __call__(self, rasync=False):
        return self.run(rasync)

    def close(self):
        """Close algorithm execution."""
        lib.Close(self.descr)

    def predict(self, data):
        """Predict """
        arr, l1, l2 = bind.to_c_2d_array(data)
        result = lib.Predict(self.descr, arr, l1, l2)
        return bind.to_managed_1d_array(result)

    @property
    def centroids(self):
        """Get centroids."""
        result = lib.RealCentroids(self.descr)
        return bind.to_managed_2d_array(result)

    def __len__(self):
        return len(self.centroids)

    def __getitem__(self, key):
        return self.centroids[key]

    def __iter__(self):
        return iter(self.centroids)

    def __contains__(self, data):
        return data in self.centroids
