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
        self._set_descr()

    def _set_descr(self):
        if hasattr(self, '_OnlineClust__finalize'):
            _, free, _, _ = self.__finalize.detach()
            free()
        descr = getattr(lib, self.__class__.__name__.upper())
        self.descr = descr(*self.args)
        self.__finalize = weakref.finalize(self, _make_free(self.descr))

    def fit(self, data):
        """Execute sequentially push, run and close methods.

        :param data: data to process"""
        self._set_descr()
        self.push(data)
        self.run()
        self.close()

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
        err = lib.Run(self.descr, 1 if rasync else 0)
        if err != 0:
            raise RuntimeError(
                'Wrong initialization parameters or is already running'
            )

        return err

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


def _make_free(descr):
    def free():
        lib.Free(descr)
    return free
