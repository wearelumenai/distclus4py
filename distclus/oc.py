import weakref

import numpy as np

from distclus.bind import handle_error
from . import bind
from .ffi import lib


class OnlineClust:
    """Base class for algorithm implementation using a native library"""

    def __init__(self, builder, space='vectors', data=None, *args):
        self.builder = builder
        space = bind.space(space)
        arr, l1, l2, l3 = bind.to_c_array(data)
        self.args = [space, arr, l1, l2, l3] + list(args)
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
        """
        Sequentially push train data, run in synchronous mode
        and close the algorithm.
        :param data: train data
        """
        self._set_descr()
        self.push(data)
        self.run()
        self.close()

    def push(self, data):
        """
        Push train data to the algorithm
        :param data: train data
        """
        arr, l1, l2, l3 = bind.to_c_array(data)
        err = lib.Push(self.descr, arr, l1, l2, l3)
        handle_error(err)

    def run(self, rasync=False):
        """
        Execute the algorithm in synchronous or asynchronous mode
        :param bool rasync: if True run in asynchronous mode
        otherwise run in synchronous mode (default)
        """
        err = lib.Run(self.descr, 1 if rasync else 0)
        handle_error(err)

    def predict(self, data):
        """
        Get labels for input data
        :param data: input data
        :return: output labels
        """
        arr, l1, l2, l3 = bind.to_c_array(data)
        result = lib.Predict(self.descr, arr, l1, l2, l3)
        handle_error(result.err)
        labels = bind.Array(addr=result.labels, l1=result.n1)
        return bind.to_managed_array(labels)

    @property
    def centroids(self):
        """Get centroids"""
        result = lib.Centroids(self.descr)
        handle_error(result.err)
        centroids = bind.Array(
            addr=result.centroids,
            l1=result.l1, l2=result.l2, l3=result.l3
        )
        return bind.to_managed_array(centroids)

    def predict_online(self, data):
        """
        Get centroids and labels for input data
        :param data: input data
        :return: centroids and output labels
        """
        arr, l1, l2, l3 = bind.to_c_array(data)
        result = lib.Predict(self.descr, arr, l1, l2, l3)
        handle_error(result.err)
        labels = bind.Array(
            addr=result.labels,
            l1=result.n1
        )
        centroids = bind.Array(
            addr=result.centroids,
            l1=result.l1, l2=result.l2, l3=result.l3
        )
        return bind.to_managed_array(centroids), bind.to_managed_array(labels)

    def close(self):
        """
        Stop the algorithm and release resources
        """
        lib.Close(self.descr)


def _make_free(descr):
    def free():
        lib.Free(descr)

    return free
