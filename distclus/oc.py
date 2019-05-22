import weakref

import numpy as np

from distclus.bind import handle_error
from . import bind
from .ffi import lib


class OnlineClust:
    """Base class for algorithm implementation using a native library"""

    def __init__(self, builder, space='vectors', data=np.empty([0, 0]), *args):
        self.builder = builder
        space = bind.space(space)
        arr, l1, l2, _ = bind.to_c_array(data)
        self.args = [space, arr, l1, l2] + list(args)
        self._set_descr()

    def _set_descr(self):
        if hasattr(self, '_OnlineClust__finalize'):
            _, free, _, _ = self.__finalize.detach()
            free()
        algo = self.builder(*self.args)
        handle_error(algo.err)
        self.descr = algo.descr
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
        arr, l1, l2, _ = bind.to_c_array(data)
        err = lib.Push(self.descr, arr, l1, l2)
        handle_error(err)

    def __radd__(self, data):
        return self.push(data)

    def run(self, rasync=False):
        """Execute (a-)synchronously the alogrithm

        :param bool rasync: Asynchronous execution if True. Default is False.
        """
        err = lib.Run(self.descr, 1 if rasync else 0)
        handle_error(err)

    def __call__(self, rasync=False):
        return self.run(rasync)

    def close(self):
        """Close algorithm execution."""
        lib.Close(self.descr)

    def predict(self, data):
        """Predict """
        arr, l1, l2, _ = bind.to_c_array(data)
        result = lib.Predict(self.descr, arr, l1, l2)
        handle_error(result.err)
        return bind.to_managed_array(result)

    @property
    def centroids(self):
        """Get centroids."""
        result = lib.Centroids(self.descr)
        handle_error(result.err)
        return bind.to_managed_array(result)


def _make_free(descr):
    def free():
        lib.Free(descr)

    return free
